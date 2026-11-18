"""分页 KV 的单元测试（纯 CPU，秒级）。

覆盖的都是"错了也不会报错，只会悄悄算错/卡死"的地方：
  - 引用计数与写时复制：漏减一次就可能释放仍在用的块
  - 链式哈希：[A][B] 与 [B][A] 必须不撞
  - 只对写满的块建索引：没满的块内容还会变
  - evict() 必须能终止（这里曾有过死循环）
  - 分页 gather 出来的 K/V 必须与连续布局逐元素相等

运行：
    python tests/test_paged_kv.py
或   pytest tests/test_paged_kv.py -q
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import torch

from llm_lab.paged_attn.block_manager import AllocStatus, BlockManager
from llm_lab.paged_attn.cache import ContiguousKVCache, PagedKVCache
from llm_lab.paged_attn.prefix_cache import PrefixCache


def test_allocate_free_roundtrip() -> None:
    bm = BlockManager(num_blocks=8, block_size=4)
    ids = bm.allocate(3)
    assert len(ids) == 3 and bm.num_free_blocks == 5
    for pid in ids:
        assert bm.blocks[pid].ref_count == 1
    bm.free(ids)
    assert bm.num_free_blocks == 8
    for pid in ids:
        assert bm.blocks[pid].ref_count == 0


def test_double_free_is_rejected() -> None:
    bm = BlockManager(num_blocks=4, block_size=4)
    ids = bm.allocate(1)
    bm.free(ids)
    try:
        bm.free(ids)
    except RuntimeError:
        return
    raise AssertionError("重复 free 应当抛错，否则引用计数会被减成负数")


def test_reference_count_and_cow() -> None:
    """两个序列共享一块，写入前必须复制出私有副本。"""
    bm = BlockManager(num_blocks=4, block_size=2)
    ids = bm.allocate(1)
    owner = ids[0]
    bm.fork(ids)  # 第二个序列共享
    assert bm.blocks[owner].ref_count == 2

    # ref_count>1 -> 必须复制
    new_id = bm.copy_on_write(owner)
    assert new_id != owner
    assert bm.blocks[new_id].ref_count == 1
    # 复制后原块引用降为 1，仍是原持有者的
    assert bm.blocks[owner].ref_count == 1

    # ref_count==1 时不该复制
    same = bm.copy_on_write(new_id)
    assert same == new_id


def test_can_allocate_status() -> None:
    bm = BlockManager(num_blocks=4, block_size=10)
    # 单条序列就要 4 块以上 -> 永远满足不了
    assert bm.can_allocate(10 * 5) is AllocStatus.NEVER
    # 池子空着 -> ok
    assert bm.can_allocate(10 * 2) is AllocStatus.OK
    bm.allocate(3)
    # 只剩 1 块，要 2 块 -> 现在不行，之后可能行
    assert bm.can_allocate(10 * 2) is AllocStatus.LATER


def test_evict_terminates() -> None:
    """回归测试：evict 曾把不可驱逐的条目放回同一个 LRU，导致 while 永不结束。"""
    bm = BlockManager(num_blocks=8, block_size=2)
    pc = PrefixCache(bm)

    # 让 4 块被"占用"（ref_count>0），它们不可驱逐
    held = pc.alloc_blocks(4)
    tokens = list(range(8))
    pc.register(held, tokens)

    # 再让另外 2 块进入索引但保持空闲（可驱逐）
    free_blocks = pc.alloc_blocks(2)
    pc.register(free_blocks, list(range(100, 104)))
    bm.free(free_blocks)

    # 请求驱逐 100 块：必须立刻返回，且最多驱逐 2 块
    freed = pc.evict(100)
    assert freed <= 2, f"只应有 2 块可驱逐，实际 {freed}"
    assert bm.blocks[held[0]].ref_count == 1, "被占用的块不该被驱逐掉"


def test_hash_is_chained() -> None:
    """[A][B] 与 [B][A] 的前缀哈希必须不同。"""
    bm = BlockManager(num_blocks=16, block_size=2)
    pc = PrefixCache(bm)
    h1 = pc.chained_hashes([1, 2, 3, 4])
    h2 = pc.chained_hashes([3, 4, 1, 2])
    assert h1 and h2
    assert h1 != h2, "块内 token 相同但顺序不同，链式哈希必须区分"

    # 相同前缀 -> 相同哈希（可复现）
    assert pc.chained_hashes([1, 2, 3, 4]) == h1
    # 只对写满的块算哈希：5 个 token / bs=2 -> 2 个完整块
    assert len(pc.chained_hashes([1, 2, 3, 4, 5])) == 2


def test_prefix_hit_shares_blocks() -> None:
    bm = BlockManager(num_blocks=16, block_size=4)
    pc = PrefixCache(bm)

    seq = list(range(100, 112))  # 12 token = 3 个满块
    first = pc.alloc_blocks(3)
    pc.register(first, seq)
    bm.free(first)  # 请求结束，块空闲但内容仍在

    hit = pc.get_computed_blocks(seq + [200])
    assert len(hit) == 3, f"应命中 3 块，实际 {len(hit)}"
    assert set(hit) == set(first), "命中的应是同一批物理块"
    for pid in hit:
        assert bm.blocks[pid].ref_count == 1, "命中后引用计数应为 1"

    # 同一个前缀再来一次 -> 共享（ref_count 变 2），不复制
    hit2 = pc.get_computed_blocks(seq + [201])
    assert hit2 == hit
    assert bm.blocks[hit[0]].ref_count == 2


def test_prefix_stops_at_first_miss() -> None:
    bm = BlockManager(num_blocks=16, block_size=4)
    pc = PrefixCache(bm)
    seq = list(range(100, 108))  # 2 块
    first = pc.alloc_blocks(2)
    pc.register(first, seq)
    bm.free(first)

    # 第 2 块内容不同 -> 只能命中 1 块
    changed = seq[:4] + [999] * 4
    hit = pc.get_computed_blocks(changed)
    assert len(hit) == 1, f"链式哈希应在第一个不匹配处停止，实际命中 {len(hit)}"


def test_realloc_purges_index() -> None:
    """块被重新分配后，旧哈希必须立刻失效，否则会复用被覆盖的 KV。

    注意构造方式：空闲队列是 FIFO，若要让它一定把"刚释放的那块"再发出来，
    必须把池子占满再整体释放（我第一版只释放一块，结果 FIFO 发出的是另一块，
    断言失败的是测试而不是被测代码）。
    """
    bm = BlockManager(num_blocks=2, block_size=2)
    pc = PrefixCache(bm)
    blocks = pc.alloc_blocks(2)
    pc.register(blocks, [11, 22, 33, 44])  # 2 个满块
    assert len(pc.hash_to_block) == 2

    bm.free(blocks)  # 两块都回到空闲队列（顺序追加）
    got = pc.alloc_blocks(1)
    assert got == [blocks[0]], f"FIFO 应先发出 {blocks[0]}，实际 {got}"
    assert len(pc.hash_to_block) == 1, "被重新分配的那块索引必须失效"
    assert blocks[0] not in pc.hash_to_block.values()


def test_fragmentation_arithmetic() -> None:
    """预分配 vs 分页的浪费率，用固定长度手工核对。"""
    lengths = [10, 16, 17, 100]
    max_len = 100
    bs = 4

    contig_alloc = len(lengths) * max_len
    paged_alloc = sum(math.ceil(n / bs) * bs for n in lengths)
    used = sum(lengths)

    assert contig_alloc == 400 and used == 143
    # 分页只多出每条序列的最后一块（最多 bs-1 个 token）
    assert paged_alloc - used < len(lengths) * bs
    assert paged_alloc < contig_alloc


def test_paged_matches_contiguous() -> None:
    """分页 gather 出来的 K/V 必须与连续布局逐元素相等，attention 输出也必须一致。"""
    torch.manual_seed(0)
    B, H, D, bs, max_len = 2, 2, 8, 4, 20
    lengths = [8, 17]

    k = torch.randn(B, max_len, H, D)
    v = torch.randn(B, max_len, H, D)

    paged = PagedKVCache(16, bs, H, D, dtype=torch.float32, device="cpu")
    contig = ContiguousKVCache(B, max_len, H, D, dtype=torch.float32, device="cpu")
    for b in range(B):
        n = lengths[b]
        paged.new_sequence(b)
        paged.append(b, k[b, :n], v[b, :n])
        contig.append(b, k[b, :n], v[b, :n])

    for b in range(B):
        n = lengths[b]
        ids = torch.tensor(paged.tables[b], dtype=torch.long)
        k_from_paged = paged._kv[:, 0][ids].reshape(-1, H, D)[:n]
        assert torch.allclose(k_from_paged, k[b, :n], atol=1e-6), f"seq {b} 的 K 顺序错了"
        kc, vc = contig.read(b)
        assert torch.allclose(kc, k[b, :n], atol=1e-6)

    # attention 输出对照（fp32，容差可以很紧）
    q = torch.randn(B, H, D)
    out_paged = []
    for b in range(B):
        n = lengths[b]
        ids = torch.tensor(paged.tables[b], dtype=torch.long)
        kb = paged._kv[:, 0][ids].reshape(-1, H, D)[:n]
        vb = paged._kv[:, 1][ids].reshape(-1, H, D)[:n]
        o = torch.nn.functional.scaled_dot_product_attention(
            q[b].unsqueeze(0).unsqueeze(2), kb.transpose(0, 1).unsqueeze(0), vb.transpose(0, 1).unsqueeze(0)
        )
        out_paged.append(o.squeeze(0).squeeze(1))
    out_paged = torch.stack(out_paged)

    out_contig = []
    for b in range(B):
        kc, vc = contig.read(b)
        o = torch.nn.functional.scaled_dot_product_attention(
            q[b].unsqueeze(0).unsqueeze(2), kc.transpose(0, 1).unsqueeze(0), vc.transpose(0, 1).unsqueeze(0)
        )
        out_contig.append(o.squeeze(0).squeeze(1))
    out_contig = torch.stack(out_contig)

    assert torch.allclose(out_paged, out_contig, atol=1e-5)


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())