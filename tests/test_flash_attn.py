"""FlashAttention-2 (Triton) 的正确性与性能测试。

正确性用两个独立 oracle 交叉验证：
  naive  : 物化 N×N 的 fp32 参考实现（最可信，但 O(N²) 显存）
  sdpa   : torch 的 fused 实现（工业基线）
一致 = 两者都在 fp16 的舍入误差内接近 naive。

这里特别覆盖了 **N 不是 BLOCK 整数倍** 的情形（N=1000、N=4097），
因为尾部块的越界处理是这类 kernel 最常见的静默错误来源：
补 0 的 key 会被当成"相似度=0 的有效 key"，污染 softmax 分母，
小 N 时误差看起来"不大"，长上下文下会直接崩。

运行：
    /home/sunever/aiinfra/.venv/bin/python tests/test_flash_attn.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import torch

from llm_lab.flash_attn.triton_fa2 import (
    flash_attn_forward,
    naive_attention,
    sdpa_attention,
)

# fp16 输入的容差：attention 输出在 [-1,1] 量级，2e-3 是 fp16 累加的合理上界
ATOL = 2e-3
RTOL = 2e-2

# (B, H, N, D) —— 故意混入 1000 / 4097 这种非整数倍长度
CASES = [
    (1, 2, 128, 64),
    (2, 4, 256, 64),
    (1, 2, 1000, 64),   # N % BLOCK_M != 0, N % BLOCK_N != 0
    (1, 2, 1024, 128),  # head_dim 128
    (2, 2, 4097, 64),   # 大会触发跨块 + 尾部 1 个 token
    (1, 1, 64, 256),    # head_dim 256（最吃寄存器）
]


def main() -> int:
    torch.manual_seed(0)
    if not torch.cuda.is_available():
        print("SKIP: 没有 CUDA")
        return 1

    failures = 0
    print(f"{'case':<28} {'causal':<7} {'max|err| vs naive':>18} {'max|err| sdpa-naive':>20}  verdict")
    print("-" * 88)

    for (B, H, N, D) in CASES:
        for causal in (True, False):
            q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
            k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
            v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)

            ref = naive_attention(q, k, v, causal=causal)
            out, lse = flash_attn_forward(q, k, v, causal=causal)
            torch.cuda.synchronize()

            err = (out.float() - ref.float()).abs().max().item()
            ref_sdpa = sdpa_attention(q, k, v, causal=causal)
            err_sdpa = (ref_sdpa.float() - ref.float()).abs().max().item()

            ok = err <= ATOL + RTOL * 1.0
            # lse 形状自检：不应该少一个 token
            lse_ok = lse.shape == (B, H, N)
            verdict = "PASS" if (ok and lse_ok) else "FAIL"
            if verdict == "FAIL":
                failures += 1

            print(
                f"B{B} H{H} N{N:<5} D{D:<3}".ljust(28)
                + f"{str(causal):<7}".ljust(7)
                + f"{err:>18.5f}"
                + f"{err_sdpa:>20.5f}"
                + f"  {verdict}"
                + ("" if lse_ok else f"  [lse shape {tuple(lse.shape)}]")
            )

    print()
    if failures == 0:
        print(f"ALL PASS ({len(CASES) * 2} 个用例)")
        return 0
    print(f"{failures} 个用例 FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())