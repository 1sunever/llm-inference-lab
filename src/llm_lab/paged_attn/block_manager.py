"""逻辑块 / 物理块 的分配器。

这是 PagedAttention 里最不"AI"、但最决定成败的一层：它把一条序列的 KV cache
从"一段连续显存"降级成"若干固定大小块的页表"，代价是查表，收益是：
  1. 内部碎片从 O(max_len - len) 降到 O(block_size - 1)，通常 block_size=16；
  2. 不同序列的相同前缀可以指向同一批物理块（见 prefix_cache.py）；
  3. 序列变长时不需要 realloc / memcpy，KV cache 不再随 batch 增长而"搬家"。

命名尽量贴近 vLLM（vllm/core/block_manager.py）：
  physical block  <-> 一块 GPU 显存里的 KV
  logical block   <-> 序列视角的第 i 个块
  block_table     <-> 逻辑块 -> 物理块 的映射，喂给 kernel 的那个 int32 张量

注意：这里刻意不引入 torch 依赖，纯 Python 数据结构，
这样单元测试和 profile 都能在 CPU 上跑（部署侧才需要 CUDA）。
"""

from __future__ import annotations

import enum
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Optional, Sequence


class AllocStatus(enum.Enum):
    """一次分配请求的三种结局，与 vLLM 的语义一致。"""

    OK = enum.auto()
    # 需要 swap/重算才能满足（显存够但被别的序列占着）
    LATER = enum.auto()
    # 永远满足不了：单条序列的上限就超过整个池子
    NEVER = enum.auto()


@dataclass
class PhysicalBlock:
    """一块 KV 显存。ref_count>0 时任何分配器都不得回收它。"""

    block_id: int
    ref_count: int = 0
    # 该块当前承载的 token id（用于前缀哈希与 copy-on-write 判断）
    token_ids: List[int] = field(default_factory=list)
    # 该块在哈希表里的键；None 表示不可被前缀复用（例如块还没写满）
    block_hash: Optional[int] = None
    # 是否已标记为"可驱逐"（ref_count==0 且写满）
    evictable: bool = False

    @property
    def free(self) -> bool:
        return self.ref_count == 0


class BlockTable:
    """一条序列的逻辑块 -> 物理块映射。

    只做两件事：查表、写表。真正的 attention kernel 拿到的就是这个 list
    转成的 int32 tensor（paged_decode.py 里用 gather 还原 K/V）。
    """

    def __init__(self) -> None:
        self._blocks: List[int] = []

    def __len__(self) -> int:
        return len(self._blocks)

    def __getitem__(self, idx: int) -> int:
        return self._blocks[idx]

    def append(self, physical_block_id: int) -> None:
        self._blocks.append(physical_block_id)

    def set_slice(self, start: int, ids: Sequence[int]) -> None:
        for i, pid in enumerate(ids):
            if start + i < len(self._blocks):
                self._blocks[start + i] = pid
            else:
                self._blocks.append(pid)

    def as_list(self) -> List[int]:
        return list(self._blocks)

    def truncate(self, n_blocks: int) -> List[int]:
        """截断并返回被摘掉的物理块 id（调用方负责 free）。"""
        dropped = self._blocks[n_blocks:]
        self._blocks = self._blocks[:n_blocks]
        return dropped


class BlockManager:
    """定长块池 + 引用计数 + 先进先出空闲队列。

    之所以用 deque 而不是 LRU 缓存：块回收顺序不影响正确性，只影响
    前缀命中率（命中率由 prefix_cache 的 LRU 驱逐策略负责），
    把两件事解耦后各自都更好测。
    """

    def __init__(self, num_blocks: int, block_size: int = 16) -> None:
        if num_blocks <= 0 or block_size <= 0:
            raise ValueError("num_blocks / block_size 必须为正")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.blocks: List[PhysicalBlock] = [
            PhysicalBlock(block_id=i) for i in range(num_blocks)
        ]
        self.free_ids: Deque[int] = deque(range(num_blocks))
        self.watermark = 0.01  # 与 vLLM 相同的保守系数：预留 1% 防抖动
        # 块被重新占用前的回调，由 PrefixCache 注入，用于摘掉它的前缀索引。
        # 为什么需要回调：allocate() 会先把 block_hash 清空再返回，
        # 调用方拿到 id 后再去"回头 purge"是拿不到哈希的（这个坑我踩过，
        # 结果是索引永不失效 -> 复用被覆盖的 KV -> 输出错但不报错）。
        self.on_reclaim: Optional[Callable[[int], None]] = None

    # ---------- 容量 ----------
    def _usable(self) -> int:
        return int(self.num_blocks * (1.0 - self.watermark))

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_ids)

    def num_blocks_for(self, num_tokens: int) -> int:
        """向上取整到块。这是"最少需要多少块"的唯一权威定义。"""
        if num_tokens <= 0:
            return 0
        return (num_tokens + self.block_size - 1) // self.block_size

    def can_allocate(self, num_tokens: int) -> AllocStatus:
        need = self.num_blocks_for(num_tokens)
        if need > self._usable():
            return AllocStatus.NEVER
        if need > self.num_free_blocks:
            return AllocStatus.LATER
        return AllocStatus.OK

    # ---------- 分配 / 回收 ----------
    def allocate(self, num_blocks: int) -> List[int]:
        """取 num_blocks 个物理块，ref_count 置 1。不够则抛错（上层先 can_allocate）。"""
        if num_blocks > self.num_free_blocks:
            raise RuntimeError(
                f"OOM: 需要 {num_blocks} 块，只有 {self.num_free_blocks} 块空闲"
            )
        ids = [self.free_ids.popleft() for _ in range(num_blocks)]
        for pid in ids:
            blk = self.blocks[pid]
            # 必须在清空 block_hash **之前**通知索引持有者
            if self.on_reclaim is not None and blk.block_hash is not None:
                self.on_reclaim(pid)
            blk.ref_count = 1
            blk.block_hash = None
            blk.evictable = False
            blk.token_ids = []
        return ids

    def free(self, block_ids: Sequence[int]) -> None:
        """减引用计数，归零才真正回到空闲队列。这是 prefix 共享能安全的前提。"""
        for pid in block_ids:
            blk = self.blocks[pid]
            if blk.ref_count == 0:
                raise RuntimeError(f"块 {pid} 重复 free（ref_count 已为 0）")
            blk.ref_count -= 1
            if blk.ref_count == 0:
                # 注意：这里不清 token_ids / block_hash。
                # 块被回收进空闲队列时内容仍有效，prefix_cache 靠这一点做命中。
                blk.evictable = True
                self.free_ids.append(pid)

    def fork(self, block_ids: Sequence[int]) -> None:
        """Copy-on-write 的前半段：只加引用，不复制。"""
        for pid in block_ids:
            self.blocks[pid].ref_count += 1

    def claim(self, block_id: int) -> None:
        """把一个当前空闲、但内容仍有效的块重新占为己有。

        这是 prefix cache 命中的落地动作：命中时块躺在空闲队列里（ref_count=0），
        但它承载的 KV 恰好就是我们要的前缀，所以不能走 allocate()（那会随机取一块
        并清空 token_ids），而是把这一块"认领"回来。
        """
        if block_id not in self.free_ids:
            raise RuntimeError(f"块 {block_id} 不在空闲队列里，无法 claim")
        # deque.remove 是 O(n)；块数在千级时完全可接受。
        # 真要上生产应换成侵入式双向链表（vLLM 的 FreeKVCacheBlockQueue 就是），
        # 这一点写在这里，免得被问"为什么不用 O(1) 结构"时答不上来。
        self.free_ids.remove(block_id)
        blk = self.blocks[block_id]
        blk.ref_count = 1
        blk.evictable = False
        # 刻意保留 token_ids 与 block_hash：前缀复用依赖内容仍然有效

    def copy_on_write(self, block_id: int) -> int:
        """ref_count>1 的块被写入前，先复制出私有副本。返回新块 id。"""
        blk = self.blocks[block_id]
        if blk.ref_count == 1:
            return block_id
        new_id = self.free_ids.popleft()
        new_blk = self.blocks[new_id]
        if self.on_reclaim is not None and new_blk.block_hash is not None:
            self.on_reclaim(new_id)
        new_blk.ref_count = 1
        new_blk.token_ids = list(blk.token_ids)
        new_blk.block_hash = None  # 副本内容独立，不再参与哈希去重
        new_blk.evictable = False
        blk.ref_count -= 1
        return new_id

    # ---------- 观测 ----------
    def stats(self) -> Dict[str, float]:
        used = self.num_blocks - self.num_free_blocks
        return {
            "num_blocks": float(self.num_blocks),
            "block_size": float(self.block_size),
            "used_blocks": float(used),
            # 真正被占用的容量（按 token 计），用来和"预分配"方案对比浪费
            "used_tokens_capacity": float(used * self.block_size),
            "utilization": used / self.num_blocks,
        }