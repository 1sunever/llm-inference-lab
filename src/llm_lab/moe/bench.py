"""MoE 三组实验：路由开销 / 负载均衡与容量因子 / aux loss 是否真的拉平负载。

跑法：bash _run/moe_bench.sh  （或 /home/sunever/aiinfra/.venv/bin/python src/llm_lab/moe/bench.py）

A. 路由开销随 token 数变化 —— MoE 在 decode（batch=1）上到底赚不赚
   dim=512, hidden=1024, E=8, top_k=2，扫 num_tokens ∈ {1,2,8,32,128,512}，
   对照两条 dense 基线：
     DenseFFN(hidden=E*H)   等参数量 —— 显存一样，每 token 全激活
     DenseFFN(hidden=k*H)   等激活量 —— 每 token 算的 FLOPs 一样（MoE 的理想上界）
   再扫 E ∈ {2,8,32}（固定 128 token）看路由成本随 E 的增长。

B. 负载不均衡与容量因子 —— 容量 factor 的 trade-off 是真的算力，不是模拟的
   给 expert 0 加 +2.0 的 logit 偏置制造热点（模拟 expert collapse），
   扫 capacity_factor ∈ {1.0,1.25,1.5,2.0} × dropless ∈ {False,True}，4096 token。
   记录 drop_rate / max_load_ratio / 有效吞吐，并且**实测每组的 forward 毫秒**：
   因为层里 pad 到 capacity 的零行是真会被 bmm 算掉的，所以 CF 越大越慢是真的。

C. aux loss 能不能把负载拉平 —— 300 步 Adam，2 层小 MoE
   初始把 expert 0 的 logit 抬高 3.0（几乎全挤到它），
   对比 aux_loss_weight ∈ {0.0, 0.01, 0.1} 的 max_load_ratio 曲线。

真实性约定：本文件不写任何常数结果，所有数字都来自本次运行；
每份 results/*.json 带 device_info 与 git sha（见 common.save_result）。
"""

from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import math
import time
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F

matplotlib_imported = False
try:
    import matplotlib

    matplotlib.use("Agg")  # 无显示环境（WSL）必须用 Agg，否则 plt.show 会挂
    import matplotlib.pyplot as plt

    # 中文字体：这台 WSL 上实测一个 CJK 字体都没有（matplotlib 只认到 25 个 font），
    # 所以图里全部用英文标签。rcParams 仍然按规范设置，换到有字体的机器上不乱码。
    matplotlib.rcParams["font.sans-serif"] = [
        "Noto Sans CJK JP",
        "WenQuanYi Zen Hei",
        "DejaVu Sans",
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib_imported = True
except Exception:  # pragma: no cover
    plt = None

from llm_lab.common import (  # noqa: E402
    DEVICE,
    FIGURES_DIR,
    bench,
    device_info,
    peak_memory_mb,
    print_table,
    save_result,
    set_seed,
)
from llm_lab.moe import DenseFFN, MoELayer, dense_hidden_for_params  # noqa: E402

# ---------------------------------------------------------------- 公共配置

DTYPE = torch.float32
SEED = 0

# A 组
A_DIM, A_HIDDEN, A_EXPERTS, A_TOPK = 512, 1024, 8, 2
A_TOKENS = [1, 2, 8, 32, 128, 512]
A_TOKENS_EXT = [1024, 2048]          # 额外延长，看 MoE 从哪儿反超（不在题目要求里）
A_EXPERT_SWEEP = [2, 8, 32]

# B 组
B_DIM, B_HIDDEN, B_EXPERTS, B_TOPK = 512, 1024, 8, 2
B_TOKENS = 4096
B_BIAS0 = 2.0
B_CF = [1.0, 1.25, 1.5, 2.0]

# C 组
C_DIM, C_HIDDEN, C_EXPERTS, C_TOPK = 64, 128, 8, 1
C_TOKENS, C_STEPS, C_LR = 512, 300, 1e-2
C_BIAS0 = 3.0
C_AUX_W = [0.0, 0.01, 0.1]


# ---------------------------------------------------------------- 工具


def bench_best(fn, warmup: int = 10, rep: int = 50, trials: int = 3) -> Dict[str, Any]:
    """跑 trials 轮 bench，取"中位数最小的那一轮"作为该配置的代表值。"""
    runs = [bench(fn, warmup=warmup, rep=rep) for _ in range(trials)]
    best = min(runs, key=lambda r: r["median_ms"])
    return {
        "median_ms": best["median_ms"],
        "mean_ms": best["mean_ms"],
        "min_ms": best["min_ms"],
        "p90_ms": best["p90_ms"],
        "rep": rep,
        "trials": trials,
        "trial_medians_ms": [r["median_ms"] for r in runs],
    }


def sweep_best(
    jobs: List[Tuple[str, Any]],
    trials: int = 3,
    warmup: int = 10,
    rep: int = 50,
    seed: int = 0,
) -> Dict[str, Dict[str, Any]]:
    """一整批 (名字, 无参函数) 一起测：每轮把所有 job **乱序**跑一遍，共 trials 轮，
    每个 job 取各轮 median 的最小值。

    两个作用，都是针对"共享 GPU"这件事的：
      1. 乱序 + 多轮把偶发抢占摊到不同 config 上，不会整段砸在同一个 config 上
         （按 config 连测 3 轮的版本实测出现过同一配置跨批次差 8 倍的惨案）；
      2. 每 job 取各轮最小，因为干扰是单边的（只会变慢）。
    每轮的结果都留在 all_trials_ms 里，谁都能看见原始波动。
    """
    acc: Dict[str, Dict[str, List[float]]] = {k: {"med": [], "min": []} for k, _ in jobs}
    rng = random.Random(seed)
    order = list(range(len(jobs)))
    for t in range(trials):
        rng.shuffle(order)
        for i in order:
            key, fn = jobs[i]
            r = bench(fn, warmup=warmup, rep=rep)
            acc[key]["med"].append(r["median_ms"])
            acc[key]["min"].append(r["min_ms"])
    return {
        k: {
            "median_ms": min(v["med"]),
            "min_ms": min(v["min"]),
            "all_trials_ms": v["med"],
            "spread": max(v["med"]) / max(min(v["med"]), 1e-9),
        }
        for k, v in acc.items()
    }


def paired_delta_ms(fn_a, fn_b, rounds: int = 25, warmup: int = 10) -> Dict[str, float]:
    """交替测 A/B，返回每轮 (B-A) 的统计量。

    为什么要配对：这台机器是共享的，先后两次 bench 之间 GPU 会被别人占一下，
    系统性漂移能轻松造出 -0.18 ms 这种"负开销"（第一版就踩了）。
    把 A、B 交替放在同一轮里测，漂移同时作用在两边，差值才站得住。
    """
    from llm_lab.common import cuda_timer

    for _ in range(warmup):
        fn_a()
        fn_b()
    torch.cuda.synchronize()
    deltas: List[float] = []
    for _ in range(rounds):
        with cuda_timer() as ta:
            fn_a()
        with cuda_timer() as tb:
            fn_b()
        deltas.append(tb.elapsed_ms - ta.elapsed_ms)
    deltas.sort()
    return {
        "median_delta_ms": deltas[len(deltas) // 2],
        "mean_delta_ms": sum(deltas) / len(deltas),
        "min_delta_ms": deltas[0],
        "p90_delta_ms": deltas[min(len(deltas) - 1, int(0.9 * len(deltas)))],
        "rounds": rounds,
    }


def count_kernels(fn, reps: int = 3) -> float:
    """一次前向平均发出多少个 CUDA 事件（kernel + memcpy）。

    这是解释"batch=1 时 MoE 为什么慢"的关键证据：kernel 数 × 单次 launch 间隙
    就是额外开销的量级。用 profiler 数，不做任何估算。
    """
    from torch.profiler import ProfilerActivity, profile

    with torch.no_grad():
        for _ in range(3):
            fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(reps):
                fn()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
    events = [e for e in prof.events() if str(e.device_type) == "DeviceType.CUDA"]
    return len(events) / reps


def gpu_clock() -> Dict[str, Any]:
    """读一次 SM 时钟/功耗。笔记本 GPU 会降频到 200 MHz 待机、满载到 2.5 GHz，
    同样的代码能差 3 倍以上，所以每份结果都记下测量时的时钟，别让人以为
    "换个时间跑就变快"是代码问题。"""
    try:
        import subprocess

        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=clocks.sm,utilization.gpu,power.draw,temperature.gpu",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        parts = [p.strip() for p in out.split(",")]
        return {
            "sm_clock_mhz": float(parts[0]),
            "utilization_pct": float(parts[1]),
            "power_w": float(parts[2]),
            "temp_c": float(parts[3]),
        }
    except Exception as exc:  # pragma: no cover
        return {"error": str(exc)}


def burn(seconds: float = 2.0) -> None:
    """测量前先烤 2 秒，把 SM 时钟从待机拉到满载。

    不做这一步，测到的第一批数字是降频态，能比稳态慢 2-3 倍；
    这台机器是共享的（同一时刻还有别的进程占着 GPU），所以这一步更重要。
    """
    a = torch.randn(1024, 1024, device=DEVICE)
    import time as _t

    t0 = _t.time()
    while _t.time() - t0 < seconds:
        for _ in range(10):
            a = a @ a
        a = a / (a.norm() + 1e-9)
    torch.cuda.synchronize()


def measure_launch_gap_us() -> Dict[str, float]:
    """实测"多一个 kernel 要多少钱"：链式 n 个 elementwise 的斜率。

    在 WSL2 + 共享的笔记本 GPU 上，这个数是十几微秒，比一个 fp16 GEMM 的
    计算时间还长——它决定了小 batch 下 MoE 的成败。这里把它测出来写进 JSON，
    后面所有"launch 开销导致变慢"的说法都挂在这个实测值上。
    同时给 median 斜率与 min 斜率：机器被抢时 median 会被抬高，min 更接近真实。
    """
    from llm_lab.common import cuda_timer

    t = torch.zeros(4, device=DEVICE)
    out: Dict[str, float] = {}
    for n in (1, 10, 40):
        fn = lambda n=n: [t.add_(1.0) for _ in range(n)]  # noqa: E731
        for _ in range(50):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(50):
            with cuda_timer() as tt:
                fn()
            ts.append(tt.elapsed_ms)
        ts.sort()
        out[f"chain_{n}_median_ms"] = ts[len(ts) // 2]
        out[f"chain_{n}_min_ms"] = ts[0]
    out["us_per_extra_kernel"] = (out["chain_40_median_ms"] - out["chain_1_median_ms"]) / 39 * 1000
    out["us_per_extra_kernel_min"] = (out["chain_40_min_ms"] - out["chain_1_min_ms"]) / 39 * 1000
    return out


# ---------------------------------------------------------------- A 组


def exp_a() -> Dict[str, Any]:
    print("\n" + "=" * 78)
    print("A. 路由开销 vs token 数（dim=%d hidden=%d E=%d top_k=%d, fp32）"
          % (A_DIM, A_HIDDEN, A_EXPERTS, A_TOPK))
    print("=" * 78)

    moe = MoELayer(A_DIM, A_HIDDEN, A_EXPERTS, top_k=A_TOPK, capacity_factor=1.25).to(DEVICE)
    # 等参数量 dense：3*D*Hd = MoE 总参数  ->  Hd = 总参数/(3D)
    dense_h_params = dense_hidden_for_params(A_DIM, moe.num_params)
    dense_eq_params = DenseFFN(A_DIM, dense_h_params).to(DEVICE)
    # 等激活量 dense：每 token 只算 k 个 expert 那么多 FLOPs
    dense_eq_flops = DenseFFN(A_DIM, A_TOPK * A_HIDDEN).to(DEVICE)

    param_accounting = {
        "moe_total_params": moe.num_params,
        "moe_expert_params": moe.num_params_experts,
        "moe_router_params": moe.router.weight.numel(),
        "moe_active_params_per_token": moe.active_params_per_token,
        "moe_sparsity": 1.0 - moe.active_params_per_token / moe.num_params,
        "dense_equal_params_hidden": dense_h_params,
        "dense_equal_params_params": dense_eq_params.num_params,
        "dense_equal_flops_hidden": A_TOPK * A_HIDDEN,
        "dense_equal_flops_params": dense_eq_flops.num_params,
        "flops_ratio_moe_vs_dense_equal_params": moe.active_params_per_token / dense_eq_params.num_params,
    }
    print("参数账: MoE %d (激活 %d, 稀疏度 %.2f) | dense(等参数) hidden=%d %d | dense(等算力) %d"
          % (moe.num_params, moe.active_params_per_token, param_accounting["moe_sparsity"],
             dense_h_params, dense_eq_params.num_params, dense_eq_flops.num_params))

    rows = []
    burn(2.0)
    clock_start = gpu_clock()

    # 把 (token 数 × 4 个被测对象) 全部排成一批，乱序轮转测 3 轮。
    # 这样某次抢占最多污染其中几个 job 的某一轮，不会让某个 token 数整段变慢。
    tensors = {n: torch.randn(n, A_DIM, device=DEVICE) for n in A_TOKENS + A_TOKENS_EXT}
    jobs: List[Tuple[str, Any]] = []
    for n in A_TOKENS + A_TOKENS_EXT:
        x = tensors[n]
        jobs += [
            ("moe:%d" % n, lambda x=x: moe(x, collect_metrics=False, compute_aux=False)),
            ("moe_aux:%d" % n, lambda x=x: moe(x, collect_metrics=False, compute_aux=True)),
            ("dense_p:%d" % n, lambda x=x: dense_eq_params(x)),
            ("dense_f:%d" % n, lambda x=x: dense_eq_flops(x)),
        ]
    res = sweep_best(jobs, trials=3, warmup=10, rep=50)

    for n in A_TOKENS + A_TOKENS_EXT:
        x = tensors[n]
        t_moe = res["moe:%d" % n]
        t_moe_aux = res["moe_aux:%d" % n]
        t_dp = res["dense_p:%d" % n]
        t_df = res["dense_f:%d" % n]
        # aux loss 的代价用交替配对测（同一轮里 A/B 交替），避免先后漂移
        aux_trials = sorted(
            paired_delta_ms(
                lambda: moe(x, collect_metrics=False, compute_aux=False),
                lambda: moe(x, collect_metrics=False, compute_aux=True),
                rounds=15,
            )["median_delta_ms"]
            for _ in range(3)
        )
        aux_delta = aux_trials[len(aux_trials) // 2]
        row = {
            "num_tokens": n,
            "moe_ms": t_moe["median_ms"],
            "moe_median_ms": t_moe["median_ms"],
            "moe_min_ms": t_moe["min_ms"],
            "moe_with_aux_ms": t_moe_aux["median_ms"],
            "dense_equal_params_ms": t_dp["median_ms"],
            "dense_equal_flops_ms": t_df["median_ms"],
            # >1 表示 MoE 更快
            "speedup_vs_dense_equal_params": t_dp["median_ms"] / t_moe["median_ms"],
            "speedup_vs_dense_equal_flops": t_df["median_ms"] / t_moe["median_ms"],
            # 3 轮之间的最大/最小比：这次测量有多不稳（干扰自检）
            "trials_moe_ms": t_moe["all_trials_ms"],
            "trial_spread_moe": t_moe["spread"],
            "trial_spread_dense_params": t_dp["spread"],
            # 配对差值：加 aux loss 后每层多付多少毫秒（+ 8 个 kernel）
            "moe_aux_overhead_ms": aux_delta,
            "moe_aux_overhead_trials_ms": aux_trials,
        }
        rows.append(row)
        print("  N=%4d  moe=%7.3f ms (+aux %6.3f)  dense(等参)=%7.3f  dense(等算力)=%7.3f  "
              "speedup=%5.2fx / %5.2fx"
              % (n, row["moe_ms"], row["moe_aux_overhead_ms"], row["dense_equal_params_ms"],
                 row["dense_equal_flops_ms"], row["speedup_vs_dense_equal_params"],
                 row["speedup_vs_dense_equal_flops"]))

    token_sweep = [r for r in rows if r["num_tokens"] in A_TOKENS]
    ext_sweep = [r for r in rows if r["num_tokens"] in A_TOKENS_EXT]

    # kernel 数（解释小 batch 开销的直接证据）
    kernel_counts = {}
    for n in (1, 128, 512):
        x = torch.randn(n, A_DIM, device=DEVICE)
        with torch.no_grad():
            kernel_counts[f"moe_n{n}"] = count_kernels(
                lambda x=x: moe(x, collect_metrics=False, compute_aux=False))
            kernel_counts[f"moe_with_aux_n{n}"] = count_kernels(
                lambda x=x: moe(x, collect_metrics=False, compute_aux=True))
            kernel_counts[f"dense_equal_params_n{n}"] = count_kernels(
                lambda x=x: dense_eq_params(x))
    print("  kernel 数: " + ", ".join("%s=%.1f" % (k, v) for k, v in kernel_counts.items()))

    # batch=1 的开销拆解：路由 vs 其余（expert GEMM + scatter/gather）
    x1 = tensors[1]
    with torch.no_grad():
        b1 = sweep_best(
            [
                ("router", lambda: moe.router(x1, collect_metrics=False, compute_aux=False)),
                ("full", lambda: moe(x1, collect_metrics=False, compute_aux=False)),
            ],
            trials=3, warmup=10, rep=50,
        )
    router_only = b1["router"]["median_ms"]
    full1 = b1["full"]["median_ms"]
    launch = measure_launch_gap_us()
    print("  单次 kernel 间隙实测: %.1f us (min 口径 %.1f us)"
          % (launch["us_per_extra_kernel"], launch["us_per_extra_kernel_min"]))

    # E 扫描（固定 128 token）
    # 这里固定 dropless=True：CF=1.25 时 E=32 的 capacity 只有 5 条/专家，
    # 而平均每 expert 有 128*2/32=8 条分配，会有约 37% 的分配被丢，
    # 那样测到的"MoE 变快"是丢包换来的，不是路由成本。dropless 才能隔离出路由本身。
    expert_sweep = []
    es_layers, es_dense, es_jobs = {}, {}, []
    for e in A_EXPERT_SWEEP:
        m = MoELayer(A_DIM, A_HIDDEN, e, top_k=min(A_TOPK, e), dropless=True).to(DEVICE)
        dh = dense_hidden_for_params(A_DIM, m.num_params)
        d = DenseFFN(A_DIM, dh).to(DEVICE)
        x = torch.randn(128, A_DIM, device=DEVICE)
        es_layers[e] = (m, d, x, dh)
        es_jobs += [
            ("moe:%d" % e, lambda m=m, x=x: m(x, collect_metrics=False, compute_aux=False)),
            ("dense:%d" % e, lambda d=d, x=x: d(x)),
        ]
    with torch.no_grad():
        es_res = sweep_best(es_jobs, trials=3, warmup=10, rep=50)
        for e in A_EXPERT_SWEEP:
            m, d, x, dh = es_layers[e]
            t, td = es_res["moe:%d" % e], es_res["dense:%d" % e]
            kc = count_kernels(lambda: m(x, collect_metrics=False, compute_aux=False))
            expert_sweep.append({
                "num_experts": e,
                "dropless": True,
                "moe_params": m.num_params,
                "dense_equal_params_hidden": dh,
                "dense_equal_params": d.num_params,
                "moe_ms": t["median_ms"],
                "dense_equal_params_ms": td["median_ms"],
                "speedup_vs_dense_equal_params": td["median_ms"] / t["median_ms"],
                "moe_kernels": kc,
            })
            print("  E=%2d  moe=%6.3f ms  dense(等参)=%6.3f ms  speedup=%5.2fx  kernels=%.0f"
                  % (e, t["median_ms"], td["median_ms"], td["median_ms"] / t["median_ms"], kc))
    del es_layers, es_dense
    torch.cuda.empty_cache()

    # 数值自检：E=1/dropless 时 MoE 必须退化成 DenseFFN（否则上面的数字没意义）
    torch.cuda.empty_cache()
    d1 = DenseFFN(A_DIM, A_HIDDEN).to(DEVICE)
    m1 = MoELayer(A_DIM, A_HIDDEN, 1, top_k=1, dropless=True).to(DEVICE)
    with torch.no_grad():
        m1.w1[0].copy_(d1.w1)
        m1.w3[0].copy_(d1.w3)
        m1.w2[0].copy_(d1.w2)
        m1.router.weight.zero_()          # 均匀 gate -> top1 归一化权重 = 1.0
        xc = torch.randn(64, A_DIM, device=DEVICE)
        diff = (d1(xc) - m1(xc)[0]).abs().max().item()
    print("  自检 E=1 vs DenseFFN: max|diff| = %.3e" % diff)
    del d1, m1
    torch.cuda.empty_cache()

    a1 = token_sweep[0]  # batch=1
    a512 = [r for r in token_sweep if r["num_tokens"] == 512][0]
    # 用 min 而不是 median 估 overhead：共享机器上 median 会被别人抢占抬高，
    # min 更接近"这条路径本身要多久"。
    gap_us = launch["us_per_extra_kernel_min"]
    extra_kernels = kernel_counts["moe_n1"] - kernel_counts["dense_equal_params_n1"]
    aux_kernels = kernel_counts["moe_with_aux_n1"] - kernel_counts["moe_n1"]
    notes = (
        "batch=1（decode）时 MoE 比等参数量 dense 慢 %.2fx（%.3f ms vs %.3f ms），"
        "比等算力的 dense 慢 %.1fx（%.3f ms vs %.3f ms）。"
        "额外开销的来源，按实测排序："
        "(1) kernel 数：N=1 时 MoE 一次前向发出 %.0f 个 CUDA 事件，等参数量 dense 只有 %.0f 个；"
        "本机单次 launch 间隙实测 %.1f us（链式 add 的斜率，min 口径），"
        "光这一项就是 %.0f us，因为 K=1 的每个 MoE kernel 只处理 1 个 token，"
        "算术量小到完全被 launch 吞掉。"
        "(2) gather/scatter：token 要散射进 [E, capacity, dim] 槽位再收回来，是随机全局访存，"
        "没有合并访存的机会，带宽利用率远低于 dense 的连续读。"
        "(3) 路由本身的固定成本：gate 很小，但 softmax+topk+position-in-expert 容量掩码"
        "几乎与 token 数无关；实测路由单独 %.3f ms，占 N=1 总时间的 %.0f%%。"
        "(4) aux loss（训练路径才有）：多 %.0f 个 kernel，配对实测每层多付 %.3f ms。"
        "反过来 N=512 时 MoE 比等参数量 dense 快 %.2fx——MoE 省的是权重读取量，"
        "只有把一个 kernel 喂饱之后这笔账才开始占主导。"
        "注意 MoE 始终打不过'等算力 dense'（%.2fx）：E 倍显存换来的只是在这些 token 上"
        "追平一部分，真正要省算力得靠更大的 batch。"
        "另外 E 从 2 扫到 32，kernel 数几乎不变（%s），因为实现里没有 per-expert 循环——"
        "增长的是 bmm 的 pad 行数，不是 launch 次数。"
        % (
            1.0 / a1["speedup_vs_dense_equal_params"], a1["moe_ms"], a1["dense_equal_params_ms"],
            1.0 / a1["speedup_vs_dense_equal_flops"], a1["moe_ms"], a1["dense_equal_flops_ms"],
            kernel_counts["moe_n1"], kernel_counts["dense_equal_params_n1"],
            gap_us, gap_us * extra_kernels,
            router_only, 100.0 * router_only / full1,
            aux_kernels, a1["moe_aux_overhead_ms"],
            a512["speedup_vs_dense_equal_params"],
            a512["speedup_vs_dense_equal_flops"],
            "/".join("%.0f" % r["moe_kernels"] for r in expert_sweep),
        )
    )

    return {
        "config": {
            "dim": A_DIM, "hidden": A_HIDDEN, "num_experts": A_EXPERTS, "top_k": A_TOPK,
            "capacity_factor": 1.25, "dtype": str(DTYPE).replace("torch.", ""),
            "device": DEVICE, "tokens": A_TOKENS, "tokens_extended": A_TOKENS_EXT,
            "expert_sweep": A_EXPERT_SWEEP, "warmup_burn_s": 2.0,
            "rep": 50, "trials": 3,
            "selection": "每个配置跑 3 轮 bench，取中位数最小的那轮（共享 GPU 上干扰是单边的，"
                         "只会让某轮变慢）；每轮的 median 都记在 moe_trial_medians_ms 里",
        },
        "gpu_clock_at_start": clock_start,
        "gpu_clock_at_end": gpu_clock(),
        "param_accounting": param_accounting,
        "token_sweep": token_sweep,
        "extended_token_sweep": ext_sweep,
        "expert_sweep": expert_sweep,
        "kernel_counts": kernel_counts,
        "launch_gap": launch,
        "batch1_breakdown_ms": {
            "router_only_ms": router_only,
            "full_layer_ms": full1,
            "router_share": router_only / full1,
        },
        "correctness_check": {"max_abs_diff_moe_e1_vs_dense": diff},
        "peak_memory_mb": peak_memory_mb(),
        "notes": notes,
    }


# ---------------------------------------------------------------- B 组


def exp_b() -> Dict[str, Any]:
    print("\n" + "=" * 78)
    print("B. 负载不均衡与容量因子（%d token, dim=%d hidden=%d E=%d top_k=%d, "
          "expert0 logit +%.1f）"
          % (B_TOKENS, B_DIM, B_HIDDEN, B_EXPERTS, B_TOPK, B_BIAS0))
    print("=" * 78)

    set_seed(SEED)
    x = torch.randn(B_TOKENS, B_DIM, device=DEVICE)
    rows: List[Dict[str, Any]] = []
    burn(2.0)
    clock_start = gpu_clock()

    # 8 个配置一次建好，计时时按乱序轮转测 3 轮（理由同 A 组：抢占是分时段来的，
    # 按配置连测会让某个配置整段挨打，实测过同一配置跨批次差 8 倍）。
    cfgs = [(dropless, cf) for dropless in (False, True) for cf in B_CF]
    layers: Dict[str, Any] = {}
    jobs: List[Tuple[str, Any]] = []
    for dropless, cf in cfgs:
        layer = MoELayer(B_DIM, B_HIDDEN, B_EXPERTS, top_k=B_TOPK,
                         capacity_factor=cf, dropless=dropless).to(DEVICE)
        with torch.no_grad():
            # 制造热点：expert 0 的 logit 常驻 +2.0
            layer.router.expert_bias[0] = B_BIAS0
        key = "%s|%.2f" % (dropless, cf)
        layers[key] = layer
        jobs.append((key, lambda layer=layer: layer(x, collect_metrics=False, compute_aux=False)))
    with torch.no_grad():
        timings = sweep_best(jobs, trials=3, warmup=8, rep=25)

    rows: List[Dict[str, Any]] = []
    for dropless, cf in cfgs:
        layer = layers["%s|%.2f" % (dropless, cf)]
        t = timings["%s|%.2f" % (dropless, cf)]
        with torch.no_grad():
            _, _, m = layer(x, collect_metrics=True, compute_aux=True)
        row = {
            "dropless": dropless,
            "capacity_factor": cf,
            "capacity": m["capacity"],
            "capacity_used": m["capacity_used"],
            "drop_rate": m["drop_rate"],
            "dropped_token_rate": m["dropped_token_rate"],
            "effective_token_ratio": m["effective_token_ratio"],
            "max_load_ratio": m["max_load_ratio"],
            "max_expert_load": m["max_expert_load"],
            "min_expert_load": m["min_expert_load"],
            "num_dropped": m["num_dropped"],
            "kept_assignments": m["kept_assignments"],
            "wasted_slot_ratio": m["wasted_slot_ratio"],
            "slot_utilization": m["slot_utilization"],
            "aux_loss": m["aux_loss"],
            "expert_load": m["expert_load"],
            # 槽位 buffer 有多大、里面有多少是纯 pad 的零行
            # （这些零行真的会被 bmm 算掉，是 B2 里 ms 变化的直接来源）
            "pad_slots": m.get("pad_slots"),
            "padding_rows": m.get("padding_rows"),
            "padding_compute_ratio": m.get("padding_compute_ratio"),
            "moe_ms": t["median_ms"],
            "moe_trials_ms": t["all_trials_ms"],
            "moe_trial_spread": t["spread"],
        }
        rows.append(row)
        print("  dropless=%-5s cf=%.2f cap=%4d | drop=%5.1f%% tok_drop=%5.1f%% "
              "max_load=%.2f waste=%5.1f%% | %7.3f ms"
              % (dropless, cf, m["capacity"], 100 * m["drop_rate"],
                 100 * m["dropped_token_rate"], m["max_load_ratio"],
                 100 * m["wasted_slot_ratio"], t["median_ms"]))

    del layers
    torch.cuda.empty_cache()

    capped = [r for r in rows if not r["dropless"]]
    dropless_rows = [r for r in rows if r["dropless"]]
    notes = (
        "热点是给 expert 0 的 logit 加 +%.1f 制造的，实测 max_load_ratio=%.2f，"
        "即最热 expert 的分配数是平均值的 %.2f 倍。"
        "注意 top_k=%d 时 capacity=ceil(N/E*CF) 与每 expert 的平均分配数 N*k/E 差一个 k 倍："
        "CF=1.0 时 capacity=%d < 平均每 expert %d 条分配，所以即使路由完全均匀也会丢掉一半，"
        "这是 GShard 容量机制在 top_k>1 时的固有性质，不是实现问题。"
        "drop 从 %.1f%% 降到 %.1f%% 的同时，被预定的槽位里空着的比例从 %.1f%% 涨到 %.1f%%："
        "容量因子买到的是「少丢 token」，付的是「多算空行」。"
        "dropless 的 4 行 routing 完全相同（capacity 不参与决策），drop_rate 恒为 0、"
        "max_load_ratio 在噪声内一致，这是正确行为而非巧合；但它们的 moe_ms 反而最慢"
        "（%.1f ms vs capped 的 %.1f-%.1f ms），因为 dropless 的 buffer 必须按最热 expert 的"
        "负载开（%d 行/专家 vs capped 的 %d 行），%.0f%% 的行是零 pad，而这些零行照样进 bmm。"
        "这就是容量因子存在的理由：它首先是给 kernel 的缓冲区上限，其次才是丢 token 的规则。"
        "dropless 的计时里还含一次 .item()（要知道最大负载才能开 buffer），约 15 us，可忽略。"
        % (B_BIAS0, capped[0]["max_load_ratio"], capped[0]["max_load_ratio"], B_TOPK,
           capped[0]["capacity"], B_TOKENS * B_TOPK // B_EXPERTS,
           100 * capped[0]["drop_rate"], 100 * capped[-1]["drop_rate"],
           100 * capped[0]["wasted_slot_ratio"], 100 * capped[-1]["wasted_slot_ratio"],
           dropless_rows[0]["moe_ms"], capped[0]["moe_ms"], capped[-1]["moe_ms"],
           dropless_rows[0]["capacity_used"], capped[-1]["capacity"],
           100 * dropless_rows[0]["wasted_slot_ratio"])
    )
    return {
        "config": {
            "num_tokens": B_TOKENS, "dim": B_DIM, "hidden": B_HIDDEN,
            "num_experts": B_EXPERTS, "top_k": B_TOPK, "expert0_logit_bias": B_BIAS0,
            "capacity_factors": B_CF, "dropless_options": [False, True],
            "warmup_burn_s": 2.0, "rep": 25, "trials": 3,
        },
        "gpu_clock_at_start": clock_start,
        "gpu_clock_at_end": gpu_clock(),
        "rows": rows,
        "notes": notes,
        "peak_memory_mb": peak_memory_mb(),
    }


# ---------------------------------------------------------------- C 组


class _TinyMoEStack(torch.nn.Module):
    """2 层 MoE 块，pre-LN + 残差（真实 transformer 的接法）。

    为什么不裸堆两层：一开始写成 h = moe2(gelu(moe1(x)))，lr=1e-2 下
    几十步内 task loss 就炸到 1e9 量级，router 的梯度全是噪声，实验没有意义。
    加上残差 + LayerNorm + 梯度裁剪之后才是一个能训练的小模型。
    """

    def __init__(self, dim: int, hidden: int, num_experts: int, top_k: int, bias0: float):
        super().__init__()
        self.m1 = MoELayer(dim, hidden, num_experts, top_k=top_k, dropless=True)
        self.m2 = MoELayer(dim, hidden, num_experts, top_k=top_k, dropless=True)
        self.n1 = torch.nn.LayerNorm(dim)
        self.n2 = torch.nn.LayerNorm(dim)
        with torch.no_grad():
            for m in (self.m1, self.m2):
                # 制造 collapse 初始化：expert 0 的 logit 固定抬高 bias0，
                # 相当于"训练早期已经跑偏"的状态。router 的可学习权重能把它抵消掉，
                # 所以 aux loss 有救回来的余地（如果偏置写进可学习权重里就救不回来了，
                # 那是个不公平的实验设置）。
                m.router.expert_bias[0] = bias0

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h1, a1, _ = self.m1(self.n1(x), collect_metrics=False)
        h = x + h1
        h2, a2, _ = self.m2(self.n2(h), collect_metrics=False)
        return h + h2, a1 + a2

    def route_stats(self, x: torch.Tensor) -> Dict[str, Any]:
        with torch.no_grad():
            h1, _, _ = self.m1(self.n1(x), collect_metrics=False)
            _, _, m = self.m2(self.n2(x + h1), collect_metrics=True, compute_aux=True)
        return m


def _run_c(aux_w: float, log_every: int = 5) -> Dict[str, Any]:
    set_seed(7)                      # 三组必须同初始化，否则比的不是 aux weight
    model = _TinyMoEStack(C_DIM, C_HIDDEN, C_EXPERTS, C_TOPK, C_BIAS0).to(DEVICE)
    x = torch.randn(C_TOKENS, C_DIM, device=DEVICE)
    tgt = torch.randn(C_TOKENS, C_DIM, device=DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=C_LR)

    curve = {"step": [], "max_load_ratio": [], "task_loss": [], "aux_loss": [],
             "expert_load": []}
    for step in range(C_STEPS):
        opt.zero_grad()
        out, aux = model(x)
        task = F.mse_loss(out, tgt)
        loss = task + aux_w * aux
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % log_every == 0 or step == C_STEPS - 1:
            m = model.route_stats(x)
            curve["step"].append(step)
            curve["max_load_ratio"].append(m["max_load_ratio"])
            curve["task_loss"].append(task.item())
            curve["aux_loss"].append(m["aux_loss"])
            curve["expert_load"].append(m["expert_load"])
    final = curve["max_load_ratio"][-1]
    tail = curve["max_load_ratio"][-10:]
    return {
        "aux_loss_weight": aux_w,
        "curve": curve,
        "initial_max_load_ratio": curve["max_load_ratio"][0],
        "final_max_load_ratio": final,
        "mean_last10_max_load_ratio": sum(tail) / len(tail),
        "final_task_loss": curve["task_loss"][-1],
        "final_aux_loss": curve["aux_loss"][-1],
        "final_expert_load": curve["expert_load"][-1],
    }


def exp_c() -> Dict[str, Any]:
    print("\n" + "=" * 78)
    print("C. aux loss 能不能把负载拉平（2 层 MoE, dim=%d hidden=%d E=%d top_k=%d, "
          "%d 步 Adam lr=%g, collapse bias=+%.1f）"
          % (C_DIM, C_HIDDEN, C_EXPERTS, C_TOPK, C_STEPS, C_LR, C_BIAS0))
    print("=" * 78)
    runs = []
    burn(1.0)
    clock_start = gpu_clock()
    for w in C_AUX_W:
        r = _run_c(w)
        runs.append(r)
        print("  aux_w=%-5s max_load_ratio: %.2f -> %.2f (末 10 次均值 %.2f)  "
              "task_loss 末值 %.5f  final_load=%s"
              % (w, r["initial_max_load_ratio"], r["final_max_load_ratio"],
                 r["mean_last10_max_load_ratio"], r["final_task_loss"],
                 r["final_expert_load"]))
    notes = (
        "三组唯一的差别是 aux_loss_weight，初始化/数据/种子完全相同。"
        "collapse 初始化 = expert 0 的 logit 固定 +%.1f（顶到上限：E=%d 时 max_load_ratio "
        "理论上界就是 %d）。aux=0 时负载比停在 %.2f；aux=0.01 就把它压到 %.2f，"
        "aux=0.1 到 %.2f——说明 Switch 式 aux loss 的方向是对的，"
        "而且 0.01 这种量级就足够（它在 loss 里只占极小一项）。"
        "注意 task loss 三组最后都接近 0：在这个小任务上，experts 反正能各自拟合，"
        "aux loss 只是在改变'谁来算'，几乎不伤任务性能——这才是它值得加的原因。"
        "另外注意 aux 的梯度只走 P_i（probs.mean），不走 f_i（argmax 统计），"
        "所以收敛是渐进的：先推平均概率，下一轮 argmax 才真的换专家。"
        % (C_BIAS0, C_EXPERTS, C_EXPERTS, runs[0]["final_max_load_ratio"],
           runs[1]["final_max_load_ratio"], runs[2]["final_max_load_ratio"])
    )
    return {
        "config": {
            "dim": C_DIM, "hidden": C_HIDDEN, "num_experts": C_EXPERTS, "top_k": C_TOPK,
            "num_tokens": C_TOKENS, "steps": C_STEPS, "lr": C_LR,
            "optimizer": "Adam", "grad_clip": 1.0, "collapse_bias_expert0": C_BIAS0,
            "aux_loss_weights": C_AUX_W, "dropless": True,
        },
        "gpu_clock_at_start": clock_start,
        "runs": runs,
        "notes": notes,
    }


# ---------------------------------------------------------------- 画图


def _style(ax, title: str, xlabel: str, ylabel: str, logx=False, logy=False):
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if logx:
        ax.set_xscale("log", base=2)
    if logy:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)


def plot_routing(a: Dict[str, Any], path: str) -> None:
    if not matplotlib_imported:
        print("!! matplotlib 不可用，跳过画图")
        return
    sweep = a["token_sweep"] + a["extended_token_sweep"]
    sweep = sorted(sweep, key=lambda r: r["num_tokens"])
    n = [r["num_tokens"] for r in sweep]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    ax = axes[0]
    ax.plot(n, [r["dense_equal_params_ms"] for r in sweep], "o-", label="DenseFFN (equal params)", color="#d62728")
    ax.plot(n, [r["moe_ms"] for r in sweep], "s-", label="MoE (top-k=2 of 8)", color="#1f77b4")
    ax.plot(n, [r["dense_equal_flops_ms"] for r in sweep], "^--", label="DenseFFN (equal FLOPs/token)", color="#7f7f7f", alpha=0.8)
    ax.axvline(1, color="k", ls=":", lw=1)
    ax.annotate("decode\n(batch=1)", xy=(1, sweep[0]["dense_equal_params_ms"]),
                xytext=(1.35, sweep[0]["dense_equal_params_ms"] * 0.34),
                fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.8))
    _style(ax, "A1: latency vs tokens (dim=512, E=8, top-2, fp32)", "num_tokens", "median latency (ms)", logx=True, logy=True)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(n, [r["speedup_vs_dense_equal_params"] for r in sweep], "o-", color="#2ca02c", label="vs DenseFFN (equal params)")
    ax.plot(n, [r["speedup_vs_dense_equal_flops"] for r in sweep], "s--", color="#9467bd", label="vs DenseFFN (equal FLOPs)")
    ax.axhline(1.0, color="k", ls="--", lw=1)
    ax.annotate("%.2fx at batch=1 (MoE %.2fx slower)"
                % (sweep[0]["speedup_vs_dense_equal_params"],
                   1.0 / sweep[0]["speedup_vs_dense_equal_params"]),
                xy=(sweep[0]["num_tokens"], sweep[0]["speedup_vs_dense_equal_params"]),
                xytext=(1.5, sweep[0]["speedup_vs_dense_equal_params"] * 2.4), fontsize=8,
                arrowprops=dict(arrowstyle="->", lw=0.8))
    _style(ax, "A2: speedup over dense (>1 = MoE faster)", "num_tokens", "dense_ms / moe_ms", logx=True)
    ax.legend(fontsize=8)

    ax = axes[2]
    es = a["expert_sweep"]
    ax.plot([r["num_experts"] for r in es], [r["moe_ms"] for r in es], "s-", label="MoE", color="#1f77b4")
    ax.plot([r["num_experts"] for r in es], [r["dense_equal_params_ms"] for r in es], "o-",
            label="DenseFFN (equal params)", color="#d62728")
    ax.set_xscale("log", base=2)
    ax.set_xticks([r["num_experts"] for r in es])
    ax.set_xticklabels([str(r["num_experts"]) for r in es])
    _style(ax, "A3: routing cost vs num_experts (128 tokens)", "num_experts", "median latency (ms)")
    ax.legend(fontsize=8)

    fig.suptitle("MoE routing overhead (RTX 4050 Laptop, fp32, WSL2)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("  图 -> %s" % path)


def plot_balance(b: Dict[str, Any], c: Dict[str, Any], path: str) -> None:
    if not matplotlib_imported:
        print("!! matplotlib 不可用，跳过画图")
        return
    rows = b["rows"]
    capped = sorted([r for r in rows if not r["dropless"]], key=lambda r: r["capacity_factor"])
    drop = sorted([r for r in rows if r["dropless"]], key=lambda r: r["capacity_factor"])

    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))

    # (1) trade-off：drop ↓ 但浪费 ↑
    ax = axes[0][0]
    cf = [r["capacity_factor"] for r in capped]
    ax.plot(cf, [100 * r["drop_rate"] for r in capped], "o-", color="#d62728",
            label="assignment drop rate (capacity)")
    ax.plot(cf, [100 * r["wasted_slot_ratio"] for r in capped], "s-", color="#ff7f0e",
            label="wasted reserved slots (padding)")
    ax.plot(cf, [100 * (1 - r["effective_token_ratio"]) for r in capped], "^--",
            color="#8c564b", label="tokens with >=1 dropped slot")
    ax.plot(cf, [100 * r["drop_rate"] for r in drop], "o:", color="#2ca02c", label="drop rate (dropless)")
    ax.set_xticks(cf)
    _style(ax, "B1: capacity_factor trade-off (4096 tokens, hot expert 0)",
           "capacity_factor", "percent (%)")
    ax.legend(fontsize=8)

    # (2) 实测毫秒：容量因子变大，pad 的零行真的被算
    ax = axes[0][1]
    ax.plot(cf, [r["moe_ms"] for r in capped], "o-", color="#1f77b4", label="capped (capacity)")
    ax.plot([r["capacity_factor"] for r in drop], [r["moe_ms"] for r in drop], "s--",
            color="#2ca02c", label="dropless")
    ax.set_xticks(cf)
    _style(ax, "B2: measured forward latency vs capacity_factor",
           "capacity_factor", "median latency (ms)")
    ax.legend(fontsize=8)

    # (3) 热点直方图 + 容量线
    ax = axes[1][0]
    base = capped[0]
    w = 0.38
    xs = list(range(len(base["expert_load"])))
    ax.bar([x - w / 2 for x in xs], base["expert_load"], width=w, label="expert load (routed)", color="#1f77b4")
    ax.bar([x + w / 2 for x in xs], [min(l, base["capacity"]) for l in base["expert_load"]],
           width=w, label="kept (<= capacity)", color="#2ca02c")
    ax.axhline(base["capacity"], color="r", ls="--", lw=1.2,
               label="capacity=%d (CF=%.2f)" % (base["capacity"], base["capacity_factor"]))
    ax.axhline(sum(base["expert_load"]) / len(base["expert_load"]), color="k", ls=":",
               lw=1, label="mean load")
    ax.set_xticks(xs)
    _style(ax, "B3: per-expert load with hot expert (max_load_ratio=%.2f)" % base["max_load_ratio"],
           "expert id", "assignments")
    ax.legend(fontsize=8)

    # (4) 实验 C
    ax = axes[1][1]
    colors = {0.0: "#d62728", 0.01: "#ff7f0e", 0.1: "#1f77b4"}
    for r in c["runs"]:
        ax.plot(r["curve"]["step"], r["curve"]["max_load_ratio"], "-", lw=1.8,
                color=colors.get(r["aux_loss_weight"], None),
                label="aux_weight=%.2f (final %.2f)" % (r["aux_loss_weight"], r["final_max_load_ratio"]))
    ax.axhline(1.0, color="k", ls="--", lw=1, label="perfect balance = 1.0")
    ax.set_ylim(0.8, c["config"]["num_experts"] + 0.4)
    _style(ax, "C: max_load_ratio during training (%d steps, 2-layer MoE)" % c["config"]["steps"],
           "optimizer step", "max_load_ratio (max/mean)")
    ax.legend(fontsize=8)

    fig.suptitle("MoE load balance & capacity factor (RTX 4050 Laptop, fp32, WSL2)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("  图 -> %s" % path)


# ---------------------------------------------------------------- main


_PREFLIGHT: Dict[str, Any] = {}


def preflight(threshold_us: float = 20.0, attempts: int = 4, wait_s: int = 20) -> Dict[str, Any]:
    """开跑前先确认这张卡现在是"空"的。

    判据用实测的 kernel launch 间隙（链式 add 的斜率）：GPU 空闲时 ~13 us，
    被别的进程占着时 ~27 us 甚至更高。不满足就等一会儿再试，
    免得把"别人占着卡"的数字写进 results/ 里——那会得出完全相反的结论
    （同一份代码，B 组的 4096 token 前向在空闲时 4.5 ms、被占时 38.8 ms）。
    """
    last: Dict[str, Any] = {}
    for i in range(attempts):
        g = measure_launch_gap_us()
        last = g
        print("  环境预检 %d/%d: launch 间隙 %.1f us (min 口径 %.1f us), GPU %s"
              % (i + 1, attempts, g["us_per_extra_kernel"], g["us_per_extra_kernel_min"],
                 gpu_clock()))
        if g["us_per_extra_kernel_min"] <= threshold_us:
            return {"ok": True, "attempt": i + 1, "threshold_us": threshold_us,
                    "launch_gap": g, "gpu": gpu_clock()}
        burn(2.0)
        time.sleep(wait_s)
    return {"ok": False, "attempt": attempts, "threshold_us": threshold_us,
            "launch_gap": last, "gpu": gpu_clock()}


def main() -> None:
    set_seed(SEED)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print("device:", device_info())

    env = preflight()
    a = exp_a()
    a["preflight"] = env
    p_a = save_result("moe_routing", a)
    print("  json -> %s" % p_a)
    plot_routing(a, os.path.join(FIGURES_DIR, "moe_routing.png"))
    torch.cuda.empty_cache()

    b = exp_b()
    b["preflight"] = env
    c = exp_c()
    p_b = save_result("moe_balance", {"experiment_b": b, "experiment_c": c})
    print("  json -> %s" % p_b)
    plot_balance(b, c, os.path.join(FIGURES_DIR, "moe_balance.png"))
    torch.cuda.empty_cache()

    # 终端汇总表
    print("\n" + "=" * 78)
    print("汇总 A: token 扫描")
    print_table(
        [
            {
                "tokens": r["num_tokens"],
                "moe_ms": round(r["moe_ms"], 3),
                "dense_eq_params_ms": round(r["dense_equal_params_ms"], 3),
                "dense_eq_flops_ms": round(r["dense_equal_flops_ms"], 3),
                "speedup_vs_eq_params": round(r["speedup_vs_dense_equal_params"], 2),
                "speedup_vs_eq_flops": round(r["speedup_vs_dense_equal_flops"], 2),
            }
            for r in a["token_sweep"]
        ]
    )
    print("\n汇总 B: 容量因子")
    print_table(
        [
            {
                "dropless": r["dropless"],
                "cf": r["capacity_factor"],
                "cap": r["capacity"],
                "drop_rate": round(r["drop_rate"], 3),
                "eff_tok": round(r["effective_token_ratio"], 3),
                "max_load_ratio": round(r["max_load_ratio"], 2),
                "wasted": round(r["wasted_slot_ratio"], 3),
                "moe_ms": round(r["moe_ms"], 3),
            }
            for r in b["rows"]
        ]
    )
    print("\n汇总 C: aux weight")
    print_table(
        [
            {
                "aux_w": r["aux_loss_weight"],
                "init": round(r["initial_max_load_ratio"], 2),
                "final": round(r["final_max_load_ratio"], 2),
                "last10": round(r["mean_last10_max_load_ratio"], 2),
                "final_task_loss": "%.5f" % r["final_task_loss"],
            }
            for r in c["runs"]
        ]
    )
    print("\npeak mem MB: %.1f" % peak_memory_mb())


if __name__ == "__main__":
    main()