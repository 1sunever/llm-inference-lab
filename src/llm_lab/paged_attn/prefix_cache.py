"""前缀缓存：把「相同的 token 前缀」映射到「同一批物理块」。

这是 PagedAttention 顺带拿到的第二个红利。第一个红利是消除碎片
（块内最多浪费 block_size-1 个 token），第二个是**跨请求复用**：
多轮对话、few-shot、RAG 里那些永远重复的前缀，只需要算一次 prefill。

机制本身很小，难点全在"什么时候能安全复用"：
  1. 哈希必须是**链式**的：block i 的哈希 = H(block i-1 的哈希, block i 的 token)。
     只哈希块内 token 会让 "A B" 和 "B A" 撞成同一个键（vLLM 也是这么做的，
     见 vllm/core/prefix_caching_block.py 的 hash_block_tokens）。
  2. 只对**写满的块**建索引：块没满说明内容还会变，复用会拿到错的 KV。
  3. 块被重新分配时索引必须立刻失效，否则复用到一个已被别的请求覆盖的块，
     结果是"输出胡言乱语但不报错"——这类 bug 最难查。
  4. ref_count > 0 的块不能被驱逐。
"""

from __future__ import annotations

import hashlib
import struct
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

from .block_manager import BlockManager


class PrefixCache:
    """块级前缀索引 + LRU 驱逐。不含任何张量，纯记账，所以能跑几千条序列的仿真。"""

    def __init__(self, block_manager: BlockManager) -> None:
        self.bm = block_manager
        self.block_size = block_manager.block_size
        # 链式哈希 -> 物理块 id
        self.hash_to_block: Dict[int, int] = {}
        # 最久未使用在前；只有 ref_count==0 的块会留在里面
        self.lru: "OrderedDict[int, None]" = OrderedDict()
        # 观测指标
        self.hit_blocks = 0
        self.query_blocks = 0

    # ------------------------------------------------------------ 哈希

    def _hash_tokens(self, prev_hash: int, tokens: Tuple[int, ...]) -> int:
        """blake2b 而不是 Python 内置 hash()：后者每次进程启动加盐，不可复现。"""
        h = hashlib.blake2b(digest_size=8)
        h.update(prev_hash.to_bytes(8, "little"))
        h.update(struct.pack(f"{len(tokens)}i", *tokens))
        return int.from_bytes(h.digest(), "little")

    def chained_hashes(self, token_ids: List[int]) -> List[int]:
        """只对写满的块算哈希；返回每个完整块的链式哈希。"""
        out: List[int] = []
        prev = 0
        n_full = len(token_ids) // self.block_size
        for i in range(n_full):
            toks = tuple(token_ids[i * self.block_size : (i + 1) * self.block_size])
            prev = self._hash_tokens(prev, toks)
            out.append(prev)
        return out

    # ------------------------------------------------------------ 命中

    def get_computed_blocks(self, token_ids: List[int]) -> List[int]:
        """返回可直接复用的物理块（**最长**前缀），并把引用计数加 1。

        遇到第一个未命中就停：链式哈希保证了后面的块即使哈希撞上也不可能
        是同一个前缀（因为它依赖前一块的哈希）。
        """
        hashes = self.chained_hashes(token_ids)
        self.query_blocks += len(hashes)
        blocks: List[int] = []
        for h in hashes:
            bid = self.hash_to_block.get(h)
            if bid is None:
                break
            if bid in self.bm.free_ids:
                # 空闲但内容还在 -> 认领回来
                self.bm.claim(bid)
            else:
                # 正被别的序列使用 -> 共享（只加引用，不复制）
                self.bm.fork([bid])
            self.lru.pop(h, None)  # 被引用中，不可驱逐
            blocks.append(bid)
        self.hit_blocks += len(blocks)
        return blocks

    # ------------------------------------------------------------ 登记

    def register(self, block_ids: List[int], token_ids: List[int]) -> None:
        """序列写完后，把它的完整块登记进索引。"""
        prev = 0
        n_full = len(token_ids) // self.block_size
        for i in range(n_full):
            if i >= len(block_ids):
                break
            toks = tuple(token_ids[i * self.block_size : (i + 1) * self.block_size])
            prev = self._hash_tokens(prev, toks)
            if prev in self.hash_to_block:
                continue
            bid = block_ids[i]
            blk = self.bm.blocks[bid]
            blk.block_hash = prev
            blk.token_ids = list(toks)
            self.hash_to_block[prev] = bid
            self.lru[prev] = None

    # ------------------------------------------------------------ 驱逐

    def evict(self, num_blocks: int) -> int:
        """按 LRU 丢掉索引项。返回实际驱逐的块数。

        注意语义：这些块的物理显存**早已**是空闲的（ref_count 归零时就回收了），
        这里丢掉的是"它的内容还值得复用"这个判断。所以驱逐降低的是命中率，
        不是显存占用——这一点在报告里必须写清，否则会被理解成两回事。
        """
        freed = 0
        # 被引用中的块不能删索引，但也不能在本次循环里立刻塞回 lru——
        # 否则 popitem 拿到的还是它，while 永远出不来（我第一版就是这么死循环的）。
        # 先攒起来，循环结束再放回。
        skipped: List[int] = []
        while freed < num_blocks and self.lru:
            h, _ = self.lru.popitem(last=False)
            bid = self.hash_to_block.get(h)
            if bid is None:
                continue
            blk = self.bm.blocks[bid]
            if blk.ref_count != 0:
                skipped.append(h)
                continue
            del self.hash_to_block[h]
            blk.block_hash = None
            freed += 1
        for h in skipped:
            self.lru[h] = None
        return freed

    # ------------------------------------------------------------ 与分配器协作

    def alloc_blocks(self, n: int) -> List[int]:
        """所有分配都走这里，而不是直接调 bm.allocate。

        因为 bm.allocate 会从 FIFO 空闲队列取块并清空内容，
        如果取到的是一块已登记哈希的块，索引必须同步失效。
        """
        ids = self.bm.allocate(n)
        for pid in ids:
            self._purge(pid)
        return ids

    def _purge(self, block_id: int) -> None:
        h = self.bm.blocks[block_id].block_hash
        if h is not None:
            self.hash_to_block.pop(h, None)
            self.lru.pop(h, None)

    # ------------------------------------------------------------ 观测

    @property
    def hit_rate(self) -> float:
        return self.hit_blocks / self.query_blocks if self.query_blocks else 0.0

    def stats(self) -> Dict[str, float]:
        return {
            "hit_blocks": float(self.hit_blocks),
            "query_blocks": float(self.query_blocks),
            "hit_rate": self.hit_rate,
            "indexed_blocks": float(len(self.hash_to_block)),
        }