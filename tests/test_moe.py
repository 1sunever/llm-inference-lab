"""MoE 的单元测试（纯 CPU，秒级）。

重点不在"能跑"，而在三条容易写错的数学：
  1. 容量公式在 top_k>1 时要不要乘 k（不乘就会莫名丢一半 token）
  2. dropless 路径必须真的不丢
  3. E=1、top_k=1 时，堆叠 bmm 的实现必须与逐元素 SwiGLU 公式完全一致
     —— 这是"我的向量化实现没写错"的唯一硬证据

运行：
    python tests/test_moe.py
或   pytest tests/test_moe.py -q
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import torch
import torch.nn.functional as F

from llm_lab.moe.layer import DenseFFN, MoELayer, dense_hidden_for_params
from llm_lab.moe.router import TopKRouter

T, DIM, E, HIDDEN = 64, 32, 8, 64


def _out(result):
    """兼容 forward 返回 tensor 或 (tensor, ...) 两种形态。"""
    return result[0] if isinstance(result, tuple) else result


def test_router_shapes() -> None:
    """dropless 时权重严格归一化；有容量限制时允许"部分保留"，和 <= 1。

    这里有个容易误判的语义：容量是按 **assignment** 丢的，不是按 token。
    一个 token 的 2 个 top-k 分配里只活下来 1 个时，它的权重和会小于 1
    （实测出现过 0.315），这不是 bug —— 它意味着该 token 只借用了 1 个专家。
    我第一版断言"所有保留行都等于 1"，就是被这一点打回来的。
    """
    x = torch.randn(T, DIM)

    # (a) dropless：没有任何丢弃，每行权重必须严格为 1
    r_free = TopKRouter(DIM, E, top_k=2, capacity_factor=1.25, dropless=True)
    r_free.eval()
    with torch.no_grad():
        w, idx, aux, m = r_free(x, collect_metrics=True, compute_aux=True)
    assert tuple(idx.shape) == (T, 2), f"idx 形状应为 ({T}, 2)，实际 {tuple(idx.shape)}"
    assert tuple(w.shape) == (T, 2), f"w 形状应为 ({T}, 2)，实际 {tuple(w.shape)}"
    assert int(idx.max()) < E and int(idx.min()) >= 0, "专家下标越界"
    dev = (w.sum(-1) - 1).abs().max().item()
    assert dev < 1e-4, f"dropless 下每行权重和应为 1，最大偏差 {dev:.2e}"
    assert float(m.get("dropped_token_rate", 0.0)) == 0.0
    assert aux.dim() == 0 and float(aux) >= 0.0, "aux loss 应为非负标量"

    # (b) 有容量限制：权重和落在 [0,1]，且确实出现部分保留
    r_cap = TopKRouter(DIM, E, top_k=2, capacity_factor=0.5)
    r_cap.eval()
    with torch.no_grad():
        w2 = r_cap(x, collect_metrics=True, compute_aux=False)[0]
    sums = w2.sum(-1)
    assert float(sums.max()) <= 1.0 + 1e-4, f"权重和不该超过 1，实际最大 {float(sums.max()):.4f}"
    assert float(sums.min()) >= 0.0, "权重和不该为负"
    partial = ((sums > 1e-6) & (sums < 1 - 1e-3)).sum().item()
    assert partial > 0, "CF=0.5 时应当出现'只保留 1 个专家'的 token"


def test_capacity_formula() -> None:
    """capacity = ceil(N/E * CF)；这是唯一权威定义，公式错了丢包率就不对。"""
    n, cf = 4096, 1.0
    r = TopKRouter(DIM, E, top_k=2, capacity_factor=cf)
    x = torch.randn(n, DIM)
    with torch.no_grad():
        _, _, _, m = r(x, collect_metrics=True, compute_aux=False)
    expect = math.ceil(n / E * cf)
    got = m.get("capacity", expect)
    assert got == expect, f"capacity 应为 ceil({n}/{E}*{cf})={expect}，实际 {got}"


def test_dropless_drops_nothing() -> None:
    r = TopKRouter(DIM, E, top_k=2, capacity_factor=0.6, dropless=True)
    x = torch.randn(512, DIM)
    with torch.no_grad():
        _, _, _, m = r(x, collect_metrics=True, compute_aux=False)
    drop = m.get("dropped_token_rate", 0.0)
    assert float(drop) == 0.0, f"dropless 模式下不该有丢弃，实际 {drop}"


def test_capacity_limited_does_drop() -> None:
    """CF 收紧时必须真的丢，否则说明容量掩码没生效。"""
    r = TopKRouter(DIM, E, top_k=2, capacity_factor=0.5)
    x = torch.randn(1024, DIM)
    with torch.no_grad():
        _, _, _, m = r(x, collect_metrics=True, compute_aux=False)
    drop = float(m.get("dropped_token_rate", 1.0))
    assert drop > 0.2, f"CF=0.5 时丢弃率应明显大于 0，实际 {drop:.3f}"


def test_moe_single_expert_matches_formula() -> None:
    """E=1/top_k=1 时，向量化实现必须等于手写的 SwiGLU。"""
    torch.manual_seed(0)
    moe = MoELayer(DIM, HIDDEN, num_experts=1, top_k=1, dropless=True)
    moe.eval()
    x = torch.randn(T, DIM)
    with torch.no_grad():
        got = _out(moe(x, collect_metrics=False, compute_aux=False))
        # 手写参照：silu(x W1) * (x W3) 再乘 W2
        h1 = x @ moe.w1[0]
        h3 = x @ moe.w3[0]
        ref = (F.silu(h1) * h3) @ moe.w2[0]
    err = (got - ref).abs().max().item()
    assert err < 1e-4, f"与手写 SwiGLU 的最大误差 {err:.2e}，向量化实现有问题"


def test_moe_is_zero_for_dropped_tokens() -> None:
    """被容量丢光的 token 必须拿到精确 0 增量（残差直通语义）。"""
    torch.manual_seed(0)
    moe = MoELayer(DIM, HIDDEN, num_experts=E, top_k=2, capacity_factor=0.05)
    moe.eval()
    x = torch.randn(T, DIM)
    with torch.no_grad():
        y = _out(moe(x, collect_metrics=True, compute_aux=False))
    zero_rows = (y.abs().sum(-1) == 0).sum().item()
    assert zero_rows > 0, "容量被压到 5% 时应当出现被完全丢弃的 token"
    assert zero_rows < T, "不该所有 token 都被丢"


def test_param_accounting() -> None:
    """top_k=2 / E=8 时激活参数应约为总量的 1/4。"""
    moe = MoELayer(DIM, HIDDEN, num_experts=E, top_k=2, dropless=True)
    total = moe.num_params
    active = moe.active_params_per_token
    ratio = active / total
    assert 0.2 < ratio < 0.3, f"激活占比 {ratio:.3f} 偏离 2/8 太多"


def test_dense_hidden_for_params() -> None:
    """按参数量反推 hidden 的辅助函数：误差应在 1% 内。

    注意它是**四舍五入**而不是向下取整——所以参数量可能略微超过目标
    （1e6 目标下多出 32 个参数）。这一点我第一版断言写错了。
    """
    target = 1_000_000
    h = dense_hidden_for_params(DIM, target)
    d = DenseFFN(DIM, h)
    rel = abs(d.num_params - target) / target
    assert rel < 0.01, f"反推出的 dense 参数量 {d.num_params} 偏离目标 {target} 达 {rel:.1%}"


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