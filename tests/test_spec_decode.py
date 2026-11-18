"""投机解码的单元测试（纯 CPU，秒级）。

三条是这个模块"错了也不报错"的重灾区：
  1. 拒绝采样少了残差修正 -> 输出分布静默偏移（看起来能跑）
  2. 加速比公式写成 1/(1-α) 而漏掉 k 的饱和项 -> 指导出错误的 k
  3. 树 mask 用 tril 代替祖先链 -> 兄弟分支互相可见，算出的不是"多算"而是"算错"

运行：
    python tests/test_spec_decode.py
或   pytest tests/test_spec_decode.py -q
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import torch

from llm_lab.spec_decode.sampling import (
    expected_accepted_tokens,
    expected_speedup,
    naive_accept_draft,
    speculative_verify,
)
from llm_lab.spec_decode.tree import ancestors, build_tree, tree_attention_mask

V = 8


def _dist(seed: int, sharp: float = 1.0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(V, generator=g) * sharp
    return torch.softmax(logits, dim=-1)


def test_expected_tokens_formula() -> None:
    # E = (1 - a^(k+1)) / (1 - a)
    assert abs(expected_accepted_tokens(0.5, 1) - 1.5) < 1e-9
    assert abs(expected_accepted_tokens(0.9, 5) - (1 - 0.9**6) / 0.1) < 1e-9
    # α=0 时每一轮只能拿到 1 个（bonus 不可得），严格等于 1
    assert abs(expected_accepted_tokens(0.0, 3) - 1.0) < 1e-9
    # k 增大时单调增，但被 1/(1-α) 卡住
    cap = 1.0 / (1 - 0.8)
    for k in (1, 3, 10, 50):
        assert expected_accepted_tokens(0.8, k) < cap + 1e-9
    assert expected_accepted_tokens(0.8, 50) > expected_accepted_tokens(0.8, 10)


def test_speedup_has_interior_optimum() -> None:
    """k 存在最优值：分子饱和、分母线性增长。扫描确认拐点在内部。"""
    alpha, c = 0.9, 0.05
    vals = [expected_speedup(alpha, k, c) for k in range(1, 65)]
    best = max(vals)
    best_k = vals.index(best) + 1
    assert 1 < best_k < 64, f"最优 k 应落在内部，实际 {best_k}"
    # 越过拐点后必须回落
    assert vals[-1] < best, "k 取到上限时应低于峰值"
    # 草稿越贵，最优 k 越小
    vals_expensive = [expected_speedup(alpha, k, 0.3) for k in range(1, 65)]
    assert vals_expensive.index(max(vals_expensive)) + 1 < best_k


def test_verify_preserves_target_distribution() -> None:
    """核心正确性：拒绝采样 + 残差修正后，输出分布必须等于 target。"""
    p = _dist(0, 1.5)
    q = _dist(1, 1.5)
    n = 20000
    g = torch.Generator().manual_seed(7)

    counts = torch.zeros(V)
    for _ in range(n):
        tok = int(torch.multinomial(q, 1, generator=g))
        res = speculative_verify(q.reshape(1, V), p.reshape(1, V), [tok], generator=g)
        assert res.num_emitted >= 1
        counts[res.tokens[0]] += 1

    emp = counts / n
    err = (emp - p).abs().max().item()
    assert err < 0.02, f"经验分布与 target 的最大偏差 {err:.4f} 过大"


def test_naive_accept_all_is_biased() -> None:
    """反例：不做验证直接接受草稿 token，分布会等于 q 而不是 p。"""
    p = _dist(0, 1.5)
    q = _dist(1, 1.5)
    n = 20000
    g = torch.Generator().manual_seed(7)

    drafts = torch.multinomial(q, n, replacement=True, generator=g)
    accepted = naive_accept_draft(drafts)
    counts = torch.bincount(accepted, minlength=V).float()
    emp = counts / n
    err = (emp - p).abs().max().item()
    assert err > 0.05, "这个反例本该明显偏离 target，若通过说明反例实现有问题"


def test_tree_dedup_and_structure() -> None:
    """[[10,20,30],[10,20,40],[10,50]] 应合并成 6 个节点（文档里的例子）。"""
    parent, tokens, depth = build_tree([[10, 20, 30], [10, 20, 40], [10, 50]])
    assert tokens == [0, 10, 20, 30, 40, 50], tokens
    assert parent == [-1, 0, 1, 2, 2, 1], parent
    assert depth == [0, 1, 2, 3, 3, 2], depth

    # 同一条链重复传入不应产生新节点
    parent2, tokens2, _ = build_tree([[10, 20, 30], [10, 20, 30]])
    assert len(tokens2) == 4, "重复链必须去重"
    assert len(parent2) == 4


def test_tree_mask_only_sees_ancestors() -> None:
    """mask 必须只放行祖先链 + 自己；兄弟分支互相不可见。"""
    parent, tokens, depth = build_tree([[10, 20, 30], [10, 20, 40], [10, 50]])
    T = len(tokens)
    mask = tree_attention_mask(parent, depth)
    assert mask.shape[-2:] == (T, T), mask.shape
    m = mask[0, 0] if mask.dim() == 4 else mask

    for i in range(T):
        visible = set(torch.nonzero(m[i]).flatten().tolist())
        expect = set(ancestors(parent, i)) | {i}
        assert visible == expect, f"位置 {i} 可见 {sorted(visible)}，应为 {sorted(expect)}"

    # 具体地：30(idx=3) 与 40(idx=4) 是兄弟，必须互不可见
    idx30 = tokens.index(30)
    idx40 = tokens.index(40)
    assert not bool(m[idx30][idx40])
    assert not bool(m[idx40][idx30])

    # 每行可见数 = depth + 1
    for i in range(T):
        assert int(m[i].sum()) == depth[i] + 1


def test_tree_reduces_positions() -> None:
    """3 条独立链共 11 个位置，合并后应少于 11。"""
    chains = [[1, 2, 3, 4], [1, 2, 9, 4], [1, 2, 3, 8]]
    _, tokens, _ = build_tree(chains)
    total = sum(len(c) for c in chains)
    assert len(tokens) < total, f"合并后 {len(tokens)} 应少于原始 {total}"


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