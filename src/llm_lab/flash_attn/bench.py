"""FlashAttention 性能基准。

三条曲线放在同一张图上，缺一条就没法说明问题：
  naive  : 物化 N×N 的 S —— 显存和时间都是 O(N²)，是"为什么需要 FA"的反证
  sdpa   : torch 内置 fused kernel —— 工业基线，打不过它就没有资格谈成果
  triton : 本项目的 FA2 —— 自有实现

同时计算 **算术强度（arithmetic intensity）**，从 roofline 角度解释拐点：
  naive  的 HBM 流量 ≈ 4·N²·D·bytes（S、P 各写一次读一次），FLOPs = 4·N²·D
          → 强度 ≈ 1 FLOP/byte，远低于 4050 的 machine balance（~20 FLOP/byte）
          → 天生 memory-bound
  FA2    的 HBM 流量 ≈ 4·N·D·bytes（Q/K/V/O 各一遍），FLOPs 不变
          → 强度 ≈ N，N 越大越 compute-bound
这就是"长序列下 FA 收益变大"的定量解释，而不是"因为它用 SRAM"这种空话。

运行：
    /home/sunever/aiinfra/.venv/bin/python src/llm_lab/flash_attn/bench.py
"""

from __future__ import annotations

import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))


import torch

from llm_lab.common import bench, print_table, save_result

# Triton 3.7 起 tl.make_block_ptr 被标记为 deprecated（仍可用）。
# 这里刻意保留 block_ptr：tensor descriptor 需要 TensorDescriptor 宿主对象，
# 会把 kernel 签名复杂化，与"讲清 tiling"的教学目标冲突。README 里已注明。
warnings.filterwarnings("ignore", category=UserWarning, module="triton")

from llm_lab.flash_attn.triton_fa2 import (  # noqa: E402
    flash_attn_forward,
    naive_attention,
    sdpa_attention,
)

DTYPE = torch.float16
SEQ_LENS = [256, 512, 1024, 2048, 4096]
HEAD_DIMS = [64, 128]
BATCH, HEADS = 2, 8
# naive 在长序列下会吃掉 4·B·H·N²·bytes 显存，超过这个长度直接跳过并记 OOM
NAIVE_MAX_N = 2048


def flops_for(b: int, h: int, n: int, d: int, causal: bool) -> float:
    """attention 前向 FLOPs：QK^T 与 PV 各 2·B·H·N²·D 次乘加 = 4·N²·D。

    因果 mask 下参与计算的位置约一半，乘 0.5（BLOCK 对齐会略多算一点，忽略）。
    """
    f = 4.0 * b * h * n * n * d
    return f * 0.5 if causal else f


def peak_delta_mb(fn) -> float:
    """单次调用的显存峰值增量（MB）。用单独一次干净调用测，避免 cache 干扰。"""
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    return (peak - base) / 1024**2


def arithmetic_intensity(n: int, d: int, causal: bool, flash: bool) -> float:
    """FLOPs / HBM bytes。fp16 每个元素 2 字节。"""
    flops = flops_for(1, 1, n, d, causal)  # 单头单 batch，比值与 B/H 无关
    if flash:
        # Q,K,V,O 各读写一遍 → 4·n·d 个元素
        bytes_moved = 4.0 * n * d * 2.0
    else:
        # S 写+读、P 写+读 → 4·n² 个元素（dominant term）
        bytes_moved = 4.0 * n * n * 2.0 + 4.0 * n * d * 2.0
    return flops / bytes_moved


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("需要 CUDA")
    torch.manual_seed(0)
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name} | sm_{props.major}{props.minor} | "
          f"{props.total_memory/1024**3:.1f} GB | {props.multi_processor_count} SMs")

    rows = []
    mem_rows = []

    for d in HEAD_DIMS:
        for n in SEQ_LENS:
            for causal in (True, False):
                q = torch.randn(BATCH, HEADS, n, d, device="cuda", dtype=DTYPE)
                k = torch.randn(BATCH, HEADS, n, d, device="cuda", dtype=DTYPE)
                v = torch.randn(BATCH, HEADS, n, d, device="cuda", dtype=DTYPE)
                flops = flops_for(BATCH, HEADS, n, d, causal)

                rep = 50 if n <= 1024 else 20

                # --- sdpa ---
                t_sdpa = bench(lambda: sdpa_attention(q, k, v, causal=causal), warmup=5, rep=rep)
                mem_sdpa = peak_delta_mb(lambda: sdpa_attention(q, k, v, causal=causal))

                # --- triton ---
                t_tri = bench(lambda: flash_attn_forward(q, k, v, causal=causal), warmup=5, rep=rep)
                # triton 的显存增量里包含 lse，单独测
                mem_tri = peak_delta_mb(lambda: flash_attn_forward(q, k, v, causal=causal))

                row = {
                    "head_dim": d,
                    "seq_len": n,
                    "causal": causal,
                    "sdpa_ms": t_sdpa["median_ms"],
                    "triton_ms": t_tri["median_ms"],
                    "sdpa_tflops": flops / (t_sdpa["median_ms"] / 1000) / 1e12,
                    "triton_tflops": flops / (t_tri["median_ms"] / 1000) / 1e12,
                    "speedup_vs_sdpa": t_sdpa["median_ms"] / t_tri["median_ms"],
                    "ai_flash": arithmetic_intensity(n, d, causal, flash=True),
                    "ai_naive": arithmetic_intensity(n, d, causal, flash=False),
                    "naive_oom": n > NAIVE_MAX_N,
                }

                # --- naive（只在小 N 上跑，否则显存直接爆）---
                if n <= NAIVE_MAX_N:
                    t_naive = bench(lambda: naive_attention(q, k, v, causal=causal), warmup=3, rep=10)
                    mem_naive = peak_delta_mb(lambda: naive_attention(q, k, v, causal=causal))
                    row.update(
                        {
                            "naive_ms": t_naive["median_ms"],
                            "naive_tflops": flops / (t_naive["median_ms"] / 1000) / 1e12,
                            "speedup_vs_naive": t_naive["median_ms"] / t_tri["median_ms"],
                            "naive_mem_mb": mem_naive,
                        }
                    )
                else:
                    row.update({"naive_ms": None, "naive_tflops": None,
                                "speedup_vs_naive": None, "naive_mem_mb": None})

                rows.append(row)
                mem_rows.append({
                    "head_dim": d, "seq_len": n, "causal": causal,
                    "naive_mb": mem_naive if n <= NAIVE_MAX_N else None,
                    "sdpa_mb": mem_sdpa, "triton_mb": mem_tri,
                })

                print(f"  d={d:<4} n={n:<5} causal={str(causal):<5} "
                      f"sdpa={t_sdpa['median_ms']:8.3f}ms  triton={t_tri['median_ms']:8.3f}ms  "
                      f"x{row['speedup_vs_sdpa']:.2f} vs sdpa"
                      + (f"  x{row['speedup_vs_naive']:.2f} vs naive" if row['speedup_vs_naive'] else "  (naive skipped)"))

                del q, k, v
                torch.cuda.empty_cache()

    print("\n=== 延迟 / 吞吐（causal=True 节选）===")
    print_table(
        [r for r in rows if r["causal"]],
        ["head_dim", "seq_len", "triton_ms", "triton_tflops", "sdpa_ms", "sdpa_tflops",
         "speedup_vs_sdpa", "speedup_vs_naive"],
    )

    print("\n=== 显存峰值增量 MB ===")
    print_table(mem_rows, ["head_dim", "seq_len", "causal", "naive_mb", "sdpa_mb", "triton_mb"])

    path = save_result("flash_attn_bench", {
        "config": {"batch": BATCH, "heads": HEADS, "dtype": "float16",
                   "seq_lens": SEQ_LENS, "head_dims": HEAD_DIMS, "naive_max_n": NAIVE_MAX_N},
        "rows": rows,
        "memory": mem_rows,
    })
    print(f"\n结果已写入 {path}")

    _plot(rows, mem_rows)


def _plot(rows, mem_rows) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from llm_lab.common import FIGURES_DIR

    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    # (1) 延迟 vs 序列长度
    ax = axes[0]
    for d, ls in ((64, "-o"), (128, "-s")):
        rs = [r for r in rows if r["causal"] and r["head_dim"] == d]
        ns = [r["seq_len"] for r in rs]
        ax.plot(ns, [r["triton_ms"] for r in rs], ls, label=f"Triton FA2 d={d}")
        ax.plot(ns, [r["sdpa_ms"] for r in rs], ls, ls="--", alpha=0.7, label=f"SDPA d={d}")
        naive_rs = [r for r in rs if r["naive_ms"]]
        if naive_rs:
            ax.plot([r["seq_len"] for r in naive_rs], [r["naive_ms"] for r in naive_rs],
                    ls, ls=":", alpha=0.6, label=f"naive d={d}")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("sequence length N")
    ax.set_ylabel("latency (ms)")
    ax.set_title("Latency vs N (causal, B=2 H=8)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)

    # (2) 达成 TFLOPs
    ax = axes[1]
    for d in (64, 128):
        rs = [r for r in rows if r["causal"] and r["head_dim"] == d]
        ax.plot([r["seq_len"] for r in rs], [r["triton_tflops"] for r in rs],
                "-o", label=f"Triton d={d}")
        ax.plot([r["seq_len"] for r in rs], [r["sdpa_tflops"] for r in rs],
                "--s", alpha=0.7, label=f"SDPA d={d}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("sequence length N")
    ax.set_ylabel("achieved TFLOP/s")
    ax.set_title("Throughput vs N")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)

    # (3) 显存
    ax = axes[2]
    for d in (64, 128):
        rs = [r for r in mem_rows if r["causal"] and r["head_dim"] == d]
        ns = [r["seq_len"] for r in rs]
        ax.plot(ns, [r["triton_mb"] for r in rs], "-o", label=f"Triton d={d}")
        ax.plot(ns, [r["sdpa_mb"] for r in rs], "--s", alpha=0.7, label=f"SDPA d={d}")
        nv = [r for r in rs if r["naive_mb"]]
        if nv:
            ax.plot([r["seq_len"] for r in nv], [r["naive_mb"] for r in nv],
                    ":^", alpha=0.6, label=f"naive d={d}")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("sequence length N")
    ax.set_ylabel("peak extra memory (MB)")
    ax.set_title("Memory: O(N^2) vs O(N)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)

    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "flash_attn.png")
    fig.savefig(out, dpi=150)
    print(f"图已保存 {out}")


if __name__ == "__main__":
    main()