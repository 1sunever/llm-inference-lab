"""两种 KV cache 组织方式，以及它们在真实工作负载下的显存账。

对比对象刻意选得公平：
  ContiguousKVCache : 给每条序列预分配 max_len 的连续显存（vLLM 之前的主流做法）。
                      实现简单、kernel 好写，代价是**保留**
                      (max_len - 实际长度) 的空间，且长度不可预测时只能取上界。
  PagedKVCache      : 按 block_size 分块，用完再给下一块，碎片只剩最后一块。

关键区别不在"能不能算对"，而在**能同时服务多少条序列**：
同样的显存，paged 能塞下的序列数通常是预分配方案的 2–4 倍，
这才是吞吐差异的来源（vLLM 论文 §1 报的 20.4%–38.2% 有效利用率就是指这件事）。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from .block_manager import BlockManager
from .prefix_cache import PrefixCache


class ContiguousKVCache:
    """预分配连续 KV：`[num_seqs, 2, max_len, num_heads, head_dim]`。

    一次性申请满，所以要按"最长可能序列"来要显存——这就是浪费的根源。
    """

    def __init__(self, num_seqs: int, max_len: int, num_heads: int,
                 head_dim: int, dtype=torch.float16, device: str = "cuda",
                 reserve: bool = True) -> None:
        self.num_seqs = num_seqs
        self.max_len = max_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device
        self.lengths = [0] * num_seqs
        self._kv: Optional[torch.Tensor] = None
        if reserve:
            self._reserve()

    def _reserve(self) -> None:
        self._kv = torch.zeros(
            self.num_seqs, 2, self.max_len, self.num_heads, self.head_dim,
            dtype=self.dtype, device=self.device,
        )

    # ---- 分配语义 ----
    def can_allocate(self, seq_id: int) -> bool:
        return self.lengths[seq_id] < self.max_len

    def allocated_tokens(self) -> int:
        """预留了多少 token 容量（与是否真的写入无关）。"""
        return self.num_seqs * self.max_len

    def used_tokens(self) -> int:
        return sum(self.lengths)

    @property
    def waste_ratio(self) -> float:
        alloc = self.allocated_tokens()
        return 1.0 - self.used_tokens() / alloc if alloc else 0.0

    def memory_bytes(self) -> int:
        return self.allocated_tokens() * 2 * self.num_heads * self.head_dim * self._elem_size()

    def _elem_size(self) -> int:
        return torch.tensor([], dtype=self.dtype).element_size()

    # ---- 读写 ----
    def append(self, seq_id: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """k, v: (n, H, D)"""
        start = self.lengths[seq_id]
        n = k.shape[0]
        assert start + n <= self.max_len, f"seq {seq_id} 超过 max_len"
        self._kv[seq_id, 0, start : start + n] = k
        self._kv[seq_id, 1, start : start + n] = v
        self.lengths[seq_id] = start + n

    def read(self, seq_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        n = self.lengths[seq_id]
        return self._kv[seq_id, 0, :n], self._kv[seq_id, 1, :n]

    def free(self, seq_id: int) -> None:
        self.lengths[seq_id] = 0


class PagedKVCache:
    """分页 KV：物理块池 + 每条序列一张 block_table。

    存储布局 `[num_blocks, 2, block_size, H, D]`，
    attention 时要么 gather 成连续（简单但多一次拷贝），
    要么直接把 block_table 喂给分页 kernel（vLLM 的做法，见 paged_decode.py）。
    """

    def __init__(self, num_blocks: int, block_size: int, num_heads: int,
                 head_dim: int, dtype=torch.float16, device: str = "cuda",
                 reserve: bool = True) -> None:
        self.bm = BlockManager(num_blocks, block_size)
        self.cache = PrefixCache(self.bm)
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device
        self._kv: Optional[torch.Tensor] = None
        # seq_id -> [物理块 id]
        self.tables: Dict[int, List[int]] = {}
        self.lengths: Dict[int, int] = {}
        if reserve:
            self._reserve()

    def _reserve(self) -> None:
        self._kv = torch.zeros(
            self.bm.num_blocks, 2, self.block_size, self.num_heads, self.head_dim,
            dtype=self.dtype, device=self.device,
        )

    # ---- 分配语义 ----
    def memory_bytes(self) -> int:
        return (
            self.bm.num_blocks * self.block_size * 2 * self.num_heads
            * self.head_dim * self._elem_size()
        )

    def _elem_size(self) -> int:
        return torch.tensor([], dtype=self.dtype).element_size()

    def allocated_tokens(self) -> int:
        """已经分出去的块能装多少 token（含最后一块的内部碎片）。"""
        used_blocks = self.bm.num_blocks - self.bm.num_free_blocks
        return used_blocks * self.block_size

    def used_tokens(self) -> int:
        return sum(self.lengths.values())

    @property
    def waste_ratio(self) -> float:
        alloc = self.allocated_tokens()
        return 1.0 - self.used_tokens() / alloc if alloc else 0.0

    def new_sequence(self, seq_id: int, prefix_tokens: Optional[List[int]] = None) -> int:
        """新建序列；如果 prefix_tokens 与缓存里的某序列前缀相同，直接复用。

        返回复用的块数（= 省掉的 prefill 块数）。
        """
        self.tables[seq_id] = []
        self.lengths[seq_id] = 0
        reused = 0
        if prefix_tokens:
            hit = self.cache.get_computed_blocks(prefix_tokens)
            if hit:
                self.tables[seq_id].extend(hit)
                self.lengths[seq_id] = len(hit) * self.block_size
                reused = len(hit)
        return reused

    def _ensure_slot(self, seq_id: int, need_len: int) -> None:
        """确保 block_table 足够长；不够就追加物理块。"""
        table = self.tables[seq_id]
        need_blocks = (need_len + self.block_size - 1) // self.block_size
        while len(table) < need_blocks:
            block_id = self.cache.alloc_blocks(1)[0]
            table.append(block_id)

    # ---- 读写 ----
    def append(self, seq_id: int, k: torch.Tensor, v: torch.Tensor) -> None:
        n = k.shape[0]
        start = self.lengths[seq_id]
        self._ensure_slot(seq_id, start + n)
        table = self.tables[seq_id]
        for i in range(n):
            pos = start + i
            bid = table[pos // self.block_size]
            off = pos % self.block_size
            self._kv[bid, 0, off] = k[i]
            self._kv[bid, 1, off] = v[i]
        self.lengths[seq_id] = start + n

    def block_table(self, seq_id: int) -> torch.Tensor:
        return torch.tensor(self.tables[seq_id], dtype=torch.int32, device=self.device)

    def read(self, seq_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """gather 回连续布局。仅用于正确性验证与朴素 kernel；
        真正的分页 kernel 不该做这一步（它等于把省下的拷贝又做回来了）。"""
        n = self.lengths[seq_id]
        idx = torch.tensor(self.tables[seq_id], dtype=torch.long, device=self.device)
        blocks = self._kv[idx]  # (nb, 2, bs, H, D)
        kv = blocks.permute(1, 0, 2, 3, 4).reshape(2, -1, self.num_heads, self.head_dim)
        return kv[0, :n], kv[1, :n]

    def free(self, seq_id: int) -> None:
        self.bm.free(self.tables.get(seq_id, []))
        self.tables.pop(seq_id, None)
        self.lengths.pop(seq_id, None)

    # ---- 观测 ----
    def stats(self) -> Dict[str, float]:
        s = self.bm.stats()
        s.update({f"prefix_{k}": v for k, v in self.cache.stats().items()})
        s["seqs"] = float(len(self.tables))
        return s