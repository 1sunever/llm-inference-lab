"""投机解码实验：分布等价性 / α 与加速比 / 真实工作负载。

跑法（仓库根目录）：
    bash _run/spec_bench.sh
或
    /home/sunever/aiinfra/.venv/bin/python src/llm_lab/spec_decode/bench.py [--skip-model]

三组实验
--------------------------------------------------------------------------------
A. 分布等价性（正确性硬证据）
   8 类离散分布 p（目标）与 q（草稿），用正确实现采 20 万次，统计经验分布；
   同时跑两个"朴素但错误"的变体当反例。指标：L1 / 最大绝对误差 / 卡方。
   正确实现的 L1 应该只受采样噪声限制（约 1e-2 量级），错误的会偏到 TV(p,q) 量级。

B. 接受率 α 与加速比（理论 vs 蒙特卡洛）
   α ∈ {0.10..0.95} × k ∈ {1,2,3,5} × cost_ratio ∈ {0.05,0.1,0.3}；
   理论曲线用 sampling.expected_speedup，模拟曲线按 α 独立伯努利接受统计每轮
   接受 token 数。额外给出每个 (α, cost_ratio) 下的最优 k —— k 越大越慢的拐点。

C. 真实工作负载
   C1 纯 CPU：用 PromptLookupDrafter 在一段 ~2000 token 的中英混合文本上
      统计"草稿命中真实续写"的接受率（oracle）。不依赖任何模型权重。
   C2 GPU：加载 Qwen2.5-1.5B-Instruct(fp16)，跑真实的"起草 -> 验证"循环，
      测 tokens/step 与端到端加速比。OOM / 装不上就写 not_run + 原因。

所有数字都来自本次运行，results/*.json 里记录了 device_info 与时间戳。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from llm_lab.common import (  # noqa: E402
    FIGURES_DIR,
    device_info,
    print_table,
    save_result,
    set_seed,
)
from llm_lab.spec_decode.draft import PromptLookupDrafter, longest_common_prefix_len  # noqa: E402
from llm_lab.spec_decode.sampling import (  # noqa: E402
    distribution_distance,
    expected_accepted_tokens,
    expected_acceptance_rate,
    expected_speedup,
    naive_accept_draft,
    naive_verify_no_correction_batch,
    optimal_k,
    speculative_verify,
    speculative_verify_batch,
)
from llm_lab.spec_decode.tree import test_tree_mask  # noqa: E402

FIG_PATH = os.path.join(FIGURES_DIR, "spec_speedup.png")
MODEL_CACHE = "/home/sunever/aiinfra/models/hub/models--Qwen--Qwen2.5-1.5B-Instruct"


# =========================================================================== #
# 实验 A：分布等价性
# =========================================================================== #

# 目标分布：明显偏斜（真实 LM 的 next-token 分布通常长这样：头很尖，长尾）
P_DIST = torch.tensor([0.40, 0.22, 0.15, 0.10, 0.06, 0.04, 0.02, 0.01], dtype=torch.float32)
# 草稿分布：形状接近但顺序不同，TV(p,q)=0.30 -> 理论上限接受率 70%
Q_DIST = torch.tensor([0.20, 0.30, 0.10, 0.05, 0.15, 0.10, 0.06, 0.04], dtype=torch.float32)


def _empirical(tokens: torch.Tensor, V: int) -> Tuple[torch.Tensor, torch.Tensor]:
    counts = torch.bincount(tokens.reshape(-1).long(), minlength=V).double()
    return counts, counts / counts.sum()


def experiment_a(n_trials: int = 200_000, n_scalar: int = 20_000, seed: int = 0) -> Dict[str, Any]:
    """正确实现 vs 两个反例，各采 n_trials 次，与真实 p 比对。"""
    p, q = P_DIST.clone(), Q_DIST.clone()
    V = int(p.numel())
    assert abs(float(p.sum()) - 1.0) < 1e-6 and abs(float(q.sum()) - 1.0) < 1e-6
    g = torch.Generator().manual_seed(seed)

    # 草稿 token 按 q 采（这就是"草稿模型会给出的 token"）
    draft = torch.multinomial(q.expand(n_trials, V).contiguous(), 1, generator=g).squeeze(1)
    x = draft.view(n_trials, 1)
    qb = q.view(1, 1, V).expand(n_trials, 1, V).contiguous()
    pb = p.view(1, 1, V).expand(n_trials, 1, V).contiguous()

    # ---- 1) 正确实现（拒绝采样 + 残差修正） ----
    t0 = time.perf_counter()
    ok = speculative_verify_batch(qb, pb, x, generator=g)
    ok_secs = time.perf_counter() - t0
    ok_tokens = ok["tokens"][:, 0]
    ok_counts, ok_freq = _empirical(ok_tokens, V)
    ok_dist = distribution_distance(ok_freq, p, counts=n_trials)
    measured_accept = float(ok["num_accepted"].double().mean())  # k=1 时这就是接受率

    # ---- 2) 反例 1：不做任何验证，直接接受草稿 ----
    all_counts, all_freq = _empirical(naive_accept_draft(x)[:, 0], V)
    all_dist = distribution_distance(all_freq, p, counts=n_trials)

    # ---- 3) 反例 2：接受规则正确，但拒绝时忘了减 q ----
    bad = naive_verify_no_correction_batch(qb, pb, x, generator=g)
    _, bad_freq = _empirical(bad["tokens"][:, 0], V)
    bad_dist = distribution_distance(bad_freq, p, counts=n_trials)

    # ---- 采样噪声下限：把正确实现的结果随机劈成两半互相比 ----
    perm = torch.randperm(n_trials, generator=g)
    _, f1 = _empirical(ok_tokens[perm[: n_trials // 2]], V)
    _, f2 = _empirical(ok_tokens[perm[n_trials // 2:]], V)
    noise_l1 = float((f1 - f2).abs().sum())

    # ---- 串行参考实现 vs 批量实现的一致性（同一算法的两种写法） ----
    t0 = time.perf_counter()
    scalar_tokens: List[int] = []
    for i in range(n_scalar):
        r = speculative_verify(q.view(1, V), p.view(1, V), [int(x[i, 0])], generator=g)
        scalar_tokens.append(r.tokens[0])
    scalar_secs = time.perf_counter() - t0
    scalar_tok = torch.tensor(scalar_tokens, dtype=torch.long)
    _, scalar_freq = _empirical(scalar_tok, V)
    scalar_dist = distribution_distance(scalar_freq, p, counts=n_scalar)
    scalar_vs_batch_l1 = float((scalar_freq - ok_freq).abs().sum())

    theory_accept = expected_acceptance_rate(p, q)
    tv = 0.5 * float((p - q).abs().sum())

    payload: Dict[str, Any] = {
        "config": {
            "V": V,
            "p": [float(v) for v in p],
            "q": [float(v) for v in q],
            "n_trials": n_trials,
            "n_scalar": n_scalar,
            "seed": seed,
            "tv_distance": tv,
            "theory_accept_rate": theory_accept,
            "chi2_dof": V - 1,
            "chi2_critical_0.05": distribution_distance(ok_freq, p, counts=n_trials).get(
                "chi2_critical_0.05"
            ),
        },
        "theory_acceptance_rate": theory_accept,
        "measured_acceptance_rate": measured_accept,
        "correct": {
            "name": "speculative_verify (rejection + residual)",
            "empirical": [float(v) for v in ok_freq],
            "counts": [int(v) for v in ok_counts],
            **ok_dist,
            "batch_seconds": ok_secs,
        },
        "naive_accept_all": {
            "name": "naive: accept draft token without verification",
            "empirical": [float(v) for v in all_freq],
            **all_dist,
        },
        "naive_no_correction": {
            "name": "naive: rejection without residual correction",
            "empirical": [float(v) for v in bad_freq],
            **bad_dist,
        },
        "noise_floor": {
            "l1_between_two_independent_halves": noise_l1,
            "half_size": n_trials // 2,
        },
        "scalar_vs_batch": {
            "n_scalar": n_scalar,
            "scalar_seconds": scalar_secs,
            "l1_scalar_vs_p": scalar_dist["l1"],
            "chi2_scalar_vs_p": scalar_dist.get("chi2"),
            "l1_scalar_vs_batch_empirical": scalar_vs_batch_l1,
        },
    }

    # 硬断言：正确实现必须贴着 p，反例必须明显偏
    assert payload["correct"]["l1"] < 0.02, f"正确实现 L1 太大: {payload['correct']['l1']}"
    assert payload["naive_accept_all"]["l1"] > 0.5, "反例 1 居然没偏，实验设置有 bug"
    assert payload["naive_no_correction"]["l1"] > 0.05, "反例 2 居然没偏，实验设置有 bug"
    assert abs(measured_accept - theory_accept) < 0.005, (
        f"实测接受率 {measured_accept:.4f} 与理论 1-TV={theory_accept:.4f} 不符"
    )
    assert scalar_dist["l1"] < 0.03, f"串行参考实现 L1 太大: {scalar_dist['l1']}"
    return payload


# =========================================================================== #
# 实验 B：α 与加速比（理论 vs 蒙特卡洛）
# =========================================================================== #

ALPHAS: List[float] = [round(0.10 + 0.05 * i, 2) for i in range(18)]  # 0.10 ... 0.95
KS: List[int] = [1, 2, 3, 5]
COST_RATIOS: List[float] = [0.05, 0.1, 0.3]


def _mc_round_tokens(alpha: float, k: int, rounds: int, g: torch.Generator) -> torch.Tensor:
    """每轮产出多少 token 的蒙特卡洛：k 个位置独立地以 α 接受，遇拒即停，末位 +1。

    cumprod 的用法：把 [rounds, k] 的伯努利掩码做 cumprod(1)，
    第一个 False 之后全变 0，求和恰好是"第一个拒绝之前的接受数"。
    """
    u = (torch.rand(rounds, k, generator=g) < alpha).double()
    accepted = u.cumprod(dim=1).sum(dim=1)
    return accepted + 1.0  # 每轮必有一个 bonus / 修正 token


def experiment_b(rounds: int = 100_000, seed: int = 1, k_max: int = 32, curve_k_max: int = 16) -> Dict[str, Any]:
    g = torch.Generator().manual_seed(seed)
    table: List[Dict[str, Any]] = []
    mc_cache: Dict[Tuple[float, int], Tuple[float, float]] = {}

    for alpha in ALPHAS:
        for k in KS:
            toks = _mc_round_tokens(alpha, k, rounds, g)
            mc_tokens = float(toks.mean())
            mc_tokens_se = float(toks.std(unbiased=True)) / math.sqrt(rounds)
            mc_cache[(alpha, k)] = (mc_tokens, mc_tokens_se)
            for c in COST_RATIOS:
                th_tokens = expected_accepted_tokens(alpha, k)
                th_speed = expected_speedup(alpha, k, c)
                mc_speed = mc_tokens / (1.0 + k * c)
                table.append(
                    {
                        "alpha": alpha,
                        "k": k,
                        "cost_ratio": c,
                        "theory_tokens_per_step": th_tokens,
                        "mc_tokens_per_step": mc_tokens,
                        "mc_tokens_se": mc_tokens_se,
                        "theory_speedup": th_speed,
                        "mc_speedup": mc_speed,
                        "rel_err": abs(mc_speed - th_speed) / th_speed,
                    }
                )

    max_rel = max(row["rel_err"] for row in table)
    max_tokens_rel = max(
        abs(cache[0] - expected_accepted_tokens(a, k)) / expected_accepted_tokens(a, k)
        for (a, k), cache in mc_cache.items()
    )

    # ---- 最优 k：给定 (α, cost_ratio)，扫描 k=1..k_max 找 argmax ----
    opt_rows: List[Dict[str, Any]] = []
    for alpha in ALPHAS:
        for c in COST_RATIOS:
            best = optimal_k(alpha, c, k_max=k_max)
            opt_rows.append(
                {
                    "alpha": alpha,
                    "cost_ratio": c,
                    "k_opt": best["k"],
                    "peak_speedup": best["speedup"],
                    "speedup_at_k1": expected_speedup(alpha, 1, c),
                    "speedup_at_kmax": expected_speedup(alpha, k_max, c),
                }
            )

    # ---- 画图用的 k -> speedup 曲线（展示拐点） ----
    curves: List[Dict[str, Any]] = []
    for alpha in (0.5, 0.7, 0.9):
        for c in COST_RATIOS:
            speeds = [expected_speedup(alpha, k, c) for k in range(1, curve_k_max + 1)]
            best = optimal_k(alpha, c, k_max=curve_k_max)
            curves.append(
                {
                    "alpha": alpha,
                    "cost_ratio": c,
                    "k": list(range(1, curve_k_max + 1)),
                    "speedup": speeds,
                    "k_opt": best["k"],
                    "peak_speedup": best["speedup"],
                }
            )

    payload = {
        "config": {
            "alpha_grid": ALPHAS,
            "k_values": KS,
            "cost_ratios": COST_RATIOS,
            "rounds": rounds,
            "seed": seed,
            "k_max_for_optimum": k_max,
        },
        "table": table,
        "max_rel_err_speedup": max_rel,
        "max_rel_err_tokens_per_step": max_tokens_rel,
        "optimal_k": {"k_max": k_max, "rows": opt_rows},
        "curves": curves,
    }
    assert max_tokens_rel < 0.02, f"MC 与理论 token 数偏差过大: {max_tokens_rel}"
    assert max_rel < 0.02, f"MC 与理论加速比偏差过大: {max_rel}"

    # 一个能直接写进报告的结论：k 太大会掉头
    drops = [
        r
        for r in opt_rows
        if r["k_opt"] < k_max and r["speedup_at_kmax"] < r["peak_speedup"] * 0.95
    ]
    payload["knee_summary"] = {
        "n_rows": len(opt_rows),
        "n_rows_where_large_k_is_worse": len(drops),
        # 挑一个最有说服力的例子：峰值加速比最高的那一行
        "example": max(drops, key=lambda r: r["peak_speedup"]) if drops else None,
        "comment": "k_opt<k_max 且 k=k_max 的加速比低于峰值 5% 以上，说明 k 已经越过拐点",
    }
    return payload


# =========================================================================== #
# 实验 C：真实工作负载
# =========================================================================== #

_LOG_LINES = [
    "2026-11-18 10:00:0{} INFO  spec_decode step={} draft=5 accepted={} tokens_per_step={} alpha=0.{}",
    "2026-11-18 10:00:1{} WARN  kv_cache blocks_used={} blocks_free={} preemptions=0 gpu_mem_mb={}",
    "2026-11-18 10:00:2{} INFO  scheduler running={} waiting={} swapped={} batch_tokens={}",
]
_JSON_BLOCK = """{{
  "step": {i},
  "draft_model": "prompt_lookup",
  "max_ngram": 3,
  "max_draft": 5,
  "accepted": {a},
  "tokens_per_step": {t},
  "acceptance_rate": 0.{al}
}}"""
_CN_PARAS = [
    "投机解码把串行解码的那堵墙拆成两段：便宜的草稿模型先猜一串 token，贵的目标模型一次并行验证。"
    "关键在于拒绝采样保证输出分布不变，所以它可以原地替换现有采样器，而不会改变模型的行为。",
    "推理系统的瓶颈往往不在算力而在显存带宽：每生成一个 token 都要把全部权重读一遍，"
    "算力单元大部分时间在等数据。投机解码用一次前向验证多个 token，把这段等待摊薄。",
    "树状草稿把多条候选合并成一棵树，公共前缀只算一次。代价是注意力掩码不再是简单的下三角，"
    "而是每个位置只能看见自己的祖先链，兄弟分支之间必须互相屏蔽。",
]
_EN_PARAS = [
    "The target model verifies a whole draft in a single forward pass, so the serial dependency "
    "chain shrinks from k plus one steps to one step. That is where the wall clock win comes from.",
    "Rejection sampling keeps the output distribution exactly equal to sampling from the target "
    "model directly, which is the only reason speculative decoding can be deployed without "
    "changing any downstream evaluation results.",
    "A prompt lookup drafter needs no draft model at all: it searches the context for the longest "
    "matching suffix and copies what followed it last time. Cheap, deterministic, and useless for "
    "open ended writing.",
]
_CODE_LINES = [
    "def speculative_verify(draft_probs, target_probs, draft_tokens):",
    "    for i, x in enumerate(draft_tokens):",
    "        ratio = min(1.0, p[i, x] / q[i, x])",
    "        if uniform() < ratio:",
    "            accepted.append(x)",
    "        else:",
    "            residual = clamp(p[i] - q[i], min=0)",
    "            return accepted + [sample(normalize(residual))]",
]


def _grow_tokens(unit: str, tokenizer, target_tokens: int, joiner: str = "\n") -> Tuple[str, List[int]]:
    """把 unit 重复到至少 target_tokens 个 token，然后截断。"""
    text = unit
    ids = tokenizer(text)["input_ids"]
    guard = 0
    while len(ids) < target_tokens and guard < 64:
        text = text + joiner + unit
        ids = tokenizer(text)["input_ids"]
        guard += 1
    return text, ids[:target_tokens]


def _build_workload(tokenizer, sizes: Dict[str, int]) -> Tuple[str, List[int], List[Dict[str, Any]]]:
    """拼一段中英混合、含重复结构的工作负载，并记录每段的 token 区间。"""
    log_unit = "\n".join(
        line.format(i, i, i % 6, 1 + i % 6, i % 10) for i, line in enumerate(_LOG_LINES * 3)
    )
    json_unit = "\n".join(
        _JSON_BLOCK.format(i=i, a=i % 6, t=1 + i % 6, al=i % 10) for i in range(6)
    )
    prose_unit = "\n".join(_EN_PARAS + _CN_PARAS)
    code_unit = "\n".join(_CODE_LINES)

    specs = [
        ("prose_mixed", prose_unit, sizes.get("prose_mixed", 600)),
        ("structured_log", log_unit, sizes.get("structured_log", 600)),
        ("json_block", json_unit, sizes.get("json_block", 400)),
        ("code", code_unit, sizes.get("code", 400)),
    ]
    parts: List[str] = []
    ids_all: List[int] = []
    spans: List[Dict[str, Any]] = []
    for name, unit, n in specs:
        text, ids = _grow_tokens(unit, tokenizer, n)
        start = len(ids_all)
        ids_all.extend(ids)
        spans.append({"name": name, "start": start, "end": len(ids_all), "tokens": len(ids)})
        parts.append(text)
    return "\n".join(parts), ids_all, spans


def _oracle_acceptance(
    ids: Sequence[int],
    k: int,
    max_ngram: int = 3,
    start: int = 32,
    lookback: int = 256,
) -> Dict[str, Any]:
    """把真实文本当"目标模型"，统计 prompt-lookup 草稿的接受率（oracle）。

    这里没有模型：target 的"输出"就是原文的下一个 token。度量的含义是
    "如果模型一字不差地复述原文，lookup 草稿能对多少个 token"，
    它是 prompt-lookup 加速比的上界。
    """
    drafter = PromptLookupDrafter(max_ngram=max_ngram, max_draft=k, lookback=lookback)
    steps = drafted = accepted = hit_steps = 0
    per_step: List[int] = []
    for i in range(start, len(ids)):
        ctx = ids[:i]
        draft = drafter.forward(ctx)
        steps += 1
        if not draft:
            per_step.append(0)
            continue
        got = longest_common_prefix_len(draft, ids[i:i + len(draft)])
        drafted += len(draft)
        accepted += got
        hit_steps += 1
        per_step.append(got)
    return {
        "k": k,
        "steps": steps,
        "hit_steps": hit_steps,
        "hit_rate": hit_steps / steps if steps else 0.0,
        "draft_tokens": drafted,
        "accepted_tokens": accepted,
        "acceptance_rate": accepted / drafted if drafted else 0.0,
        "tokens_per_step": (accepted + steps) / steps if steps else 0.0,
        "per_step_accepted": per_step,
    }


def _rolling(values: Sequence[float], window: int) -> List[float]:
    out: List[float] = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        chunk = values[lo:i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def experiment_c_cpu(ids: List[int], spans: List[Dict[str, Any]], ks: Sequence[int] = (3, 5)) -> Dict[str, Any]:
    """C1：纯 CPU 的 prompt-lookup 接受率（不加载任何模型）。"""
    overall: Dict[str, Any] = {}
    curve_x: List[int] = []
    curve_y: List[float] = []
    for k in ks:
        res = _oracle_acceptance(ids, k)
        per = res.pop("per_step_accepted")
        rolled = _rolling(per, window=50)
        step = max(1, len(rolled) // 60)
        curve_x = list(range(32, len(ids)))[::step]
        curve_y = rolled[::step]
        res["smooth_window"] = 50
        res["curve_x"] = curve_x
        res["curve_y"] = curve_y
        res["rolling_mean_accepted"] = sum(per) / len(per)
        overall[f"k{k}"] = res

    per_segment: List[Dict[str, Any]] = []
    for sp in spans:
        seg = ids[sp["start"]:sp["end"]]
        if len(seg) < 64:
            continue
        r = _oracle_acceptance(seg, ks[0], start=32)
        r.pop("per_step_accepted", None)
        r.pop("curve_x", None)
        r.pop("curve_y", None)
        per_segment.append({"segment": sp["name"], "tokens": sp["tokens"], **r})

    return {
        "status": "ran",
        "workload": {
            "total_tokens": len(ids),
            "kind": "synthetic mixed zh/en text (prose + structured logs + json + code)",
            "segments": spans,
        },
        "overall": overall,
        "by_segment_k": ks[0],
        "by_segment": per_segment,
        "note": (
            "oracle 接受率：把原文当作目标模型的输出，度量 prompt-lookup 草稿能对多少个 token。"
            "它是 prompt-lookup 加速比的上界，不含任何模型权重。"
        ),
    }


# --------------------------------------------------------------------------- #
# C2：真实模型（Qwen2.5-1.5B-Instruct, fp16, 单卡 6GB）
# --------------------------------------------------------------------------- #


def _resolve_model_dir(path: str) -> str:
    """HF cache 目录 -> 真正的 snapshot 目录（含 config.json）。"""
    if os.path.exists(os.path.join(path, "config.json")):
        return path
    snap_root = os.path.join(path, "snapshots")
    if os.path.isdir(snap_root):
        for name in sorted(os.listdir(snap_root)):
            cand = os.path.join(snap_root, name)
            if os.path.exists(os.path.join(cand, "config.json")):
                return cand
    return path


class _HFEngine:
    """最小的 HF 解码引擎：prefill / 单步 / KV cache 裁剪。

    只做实验需要的事，不引入 generate()，因为我们要在循环里插入验证逻辑。
    """

    def __init__(self, model_dir: str, dtype=torch.float16) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.cuda.empty_cache()
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = (
            AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype, low_cpu_mem_usage=True)
            .to("cuda")
            .eval()
        )
        self.cache = None
        self.last_logits: Optional[torch.Tensor] = None
        self.pos = 0
        self.vocab = int(self.model.config.vocab_size)

    @torch.no_grad()
    def prefill(self, ids: Sequence[int]) -> torch.Tensor:
        out = self.model(torch.tensor([list(ids)], device="cuda"), use_cache=True)
        self.cache = out.past_key_values
        self.pos = len(ids)
        self.last_logits = out.logits[0, -1].float()
        return self.last_logits

    @torch.no_grad()
    def step(self, tokens: Sequence[int]) -> torch.Tensor:
        t = torch.tensor([list(tokens)], device="cuda")
        out = self.model(t, past_key_values=self.cache, use_cache=True)
        self.cache = out.past_key_values
        self.pos += len(tokens)
        logits = out.logits[0].float()
        self.last_logits = logits[-1]
        return logits

    def crop_to(self, length: int) -> None:
        drop = self.pos - int(length)
        if drop > 0:
            # transformers>=4.44 的 DynamicCache 用负数表示"从尾部删掉多少个"
            self.cache.crop(-drop)
            self.pos = int(length)

    def sample(self, logits: torch.Tensor, generator: torch.Generator, temperature: float = 1.0) -> int:
        probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
        return int(torch.multinomial(probs, 1, generator=generator).item())


def _run_baseline(
    engine: _HFEngine, prompt_ids: List[int], n_new: int, seed: int, temperature: float = 1.0
) -> Dict[str, Any]:
    """自回归基线：一个 token 一次前向，计时只覆盖解码循环（prefill 两边都不计）。"""
    g = torch.Generator(device="cuda").manual_seed(seed)
    engine.prefill(prompt_ids)
    generated: List[int] = []
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    while len(generated) < n_new:
        tok = engine.sample(engine.last_logits, g, temperature)
        generated.append(tok)
        engine.step([tok])
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end)
    return {
        "name": "baseline_ar",
        "k": 1,
        "steps": len(generated),
        "tokens": len(generated),
        "tokens_per_step": 1.0,
        "accepted_draft": 0,
        "draft_tokens": 0,
        "acceptance_rate": 0.0,
        "elapsed_ms": ms,
        "ms_per_token": ms / max(1, len(generated)),
    }


def _run_speculative(
    engine: _HFEngine,
    prompt_ids: List[int],
    n_new: int,
    k: int,
    seed: int,
    temperature: float = 1.0,
    max_ngram: int = 3,
) -> Dict[str, Any]:
    """完整循环：prompt-lookup 起草 -> 目标模型一次前向验证 -> 拒绝采样 -> 裁剪 KV。

    每轮的 target 成本：1 次 k-token 前向（验证）+ 1 次 1-token 前向
    （把最后一个输出 token 喂进去，恢复下一步的 logits）。后者在整个循环里是
    固定开销，真实实现也一样，所以不能省掉不计。
    """
    drafter = PromptLookupDrafter(max_ngram=max_ngram, max_draft=k, lookback=None)
    g = torch.Generator(device="cuda").manual_seed(seed)
    engine.prefill(prompt_ids)
    generated: List[int] = []
    context = list(prompt_ids)
    steps = drafted = accepted = 0
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    while len(generated) < n_new:
        draft = drafter.forward(context + generated)[:k]
        if not draft:
            tok = engine.sample(engine.last_logits, g, temperature)
            generated.append(tok)
            engine.step([tok])
            steps += 1
            continue

        kd = len(draft)
        p_first = engine.last_logits          # 预测 draft[0]
        logits = engine.step(draft)           # [kd, V]，验证 kd 个位置
        # 各位置的目标分布：p_first, logits[0..kd-2]，再加一个 bonus 行 logits[kd-1]
        p_all = torch.cat([p_first[None], logits[:-1], logits[-1:]], dim=0)
        p_probs = torch.softmax(p_all / max(temperature, 1e-6), dim=-1)

        # 草稿分布是 one-hot：q(x)=1 -> 接受概率 = p(x)
        q_probs = torch.zeros(kd, engine.vocab, dtype=torch.float32, device="cuda")
        q_probs[torch.arange(kd, device="cuda"), torch.tensor(draft, device="cuda")] = 1.0

        res = speculative_verify(q_probs, p_probs, draft, generator=g)
        emitted = res.tokens
        drafted += kd
        accepted += res.num_accepted

        # 把被拒掉的草稿 KV 剪掉，再补最后一个输出 token 的前向
        keep = engine.pos - kd + res.num_accepted
        engine.crop_to(keep)
        engine.step([emitted[-1]])
        generated.extend(emitted)
        steps += 1
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end)
    n_out = len(generated)
    return {
        "name": f"prompt_lookup_k{k}",
        "k": k,
        "steps": steps,
        "tokens": n_out,
        "tokens_per_step": n_out / steps,
        "accepted_draft": accepted,
        "draft_tokens": drafted,
        "acceptance_rate": accepted / drafted if drafted else 0.0,
        "hit_rate": drafter.hit_rate(),
        "elapsed_ms": ms,
        "ms_per_token": ms / max(1, min(n_new, n_out)),
    }


def _measure_forward_cost(
    engine: _HFEngine,
    prompt_ids: List[int],
    lengths: Sequence[int] = (1, 2, 3, 5),
    calls: int = 10,
    rounds: int = 3,
) -> Dict[str, Any]:
    """量出"一次前向处理 L 个 token"的摊销耗时，以及各长度相对 L=1 的比值。

    三个坑：
      1. 逐次调用 + 每次一个 CUDA event 会把 launch/sync 开销算进单次耗时
         （实测虚高 ~40%），所以一轮跑 `calls` 次、再除以次数；
      2. 每次调用后把 KV cache 裁回原位，保证所有长度都在同一上下文长度下测；
      3. 这台卡和别人共享，几秒内吞吐就能飘 30%。所以各长度**交错**多测几轮、
        每档取最小值——min 才是"没被抢"时的真实成本，比值也才稳定。

    这个曲线决定了实验 B 里 cost_ratio 该怎么填：小模型短上下文时，5 个 token 的
    一次前向只比 1 个 token 贵 10~20%（都是读一遍权重，算术强度很低），所以
    "验证几乎免费"在小模型上基本成立 —— 但也正是它让 k 不能太大：
    每多猜一个 token 都要多付一次这样的前向。
    """
    engine.prefill(prompt_ids)
    pos0 = engine.pos
    tok = int(prompt_ids[-1])
    best: Dict[int, float] = {int(L): float("inf") for L in lengths}

    for _ in range(int(rounds)):
        for L in lengths:
            L = int(L)
            engine.crop_to(pos0)
            for _ in range(2):  # warmup，让 allocator 稳定
                engine.step([tok] * L)
            engine.crop_to(pos0)
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(True), torch.cuda.Event(True)
            start.record()
            for _ in range(int(calls)):
                engine.step([tok] * L)
            end.record()
            torch.cuda.synchronize()
            per_call = start.elapsed_time(end) / int(calls)
            best[L] = min(best[L], per_call)

    costs = {str(L): best[L] for L in sorted(best)}
    ratios = {str(L): best[L] / best[1] for L in sorted(best)}
    return {
        "pass_ms_by_length": costs,
        "ratio_by_length": ratios,
        "single_token_ms": costs["1"],
        "k_token_pass_ms": costs[str(max(best))],
        "k": int(max(best)),
        "ratio_k_over_1": ratios[str(max(best))],
        "calls_per_round": int(calls),
        "rounds": int(rounds),
        "note": (
            "摊销计时（每轮 calls 次，减掉 event 开销），各长度交错测 rounds 轮取最小值；"
            "ratio_by_length 已归一化到 L=1，对整机吞吐漂移不敏感"
        ),
    }


def _pass_ratio(forward_cost: Dict[str, Any], length: float) -> float:
    """按实测比值曲线线性插值出"一次前向处理 length 个 token / 处理 1 个 token"的成本比。"""
    pts = sorted((int(k), float(v)) for k, v in forward_cost["ratio_by_length"].items())
    if length <= pts[0][0]:
        return pts[0][1]
    if length >= pts[-1][0]:
        return pts[-1][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= length <= x1:
            t = (length - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return pts[-1][1]


def experiment_c_gpu(
    prompt_ids: List[int],
    open_prompt_ids: List[int],
    n_new: int = 64,
    ks: Sequence[int] = (3, 5),
    reps: int = 3,
    temperature: float = 1.0,
) -> Dict[str, Any]:
    """C2：加载真实模型，跑基线 vs 投机解码，测真实加速比。

    这台卡是 6GB 笔记本 GPU，和其它进程共享，单次计时会有 ±30% 的抖动
    （实测同一配置两次能差 1.4 倍）。所以：
      * 基线 / 各档 k **交错**跑（每个 rep 里依次跑一遍），让热漂移和外部占用
        均匀落到所有配置上，而不是只砸中其中某一个；
      * 每个配置取多次 rep 里 **最快** 的那次（min 对偶发抢占最鲁棒），
        并把所有 rep 的耗时写进 JSON，抖动本身也是数据；
      * 再用隔离测出来的 forward 成本算出 predicted_speedup 作为噪声无关的交叉验证。
    """
    result: Dict[str, Any] = {"status": "not_run", "reason": ""}
    model_dir = _resolve_model_dir(MODEL_CACHE)
    if not os.path.exists(os.path.join(model_dir, "config.json")):
        result["reason"] = f"模型目录不存在或不完整: {model_dir}"
        return result

    torch.cuda.empty_cache()
    t0 = time.perf_counter()
    engine = _HFEngine(model_dir)
    load_secs = time.perf_counter() - t0
    result.update(
        {
            "status": "ran",
            "model": "Qwen2.5-1.5B-Instruct",
            "model_dir": model_dir,
            "dtype": "float16",
            "device": torch.cuda.get_device_name(0),
            "load_seconds": load_secs,
            "vram_after_load_gb": round(torch.cuda.memory_allocated() / 1024**3, 3),
            "vram_peak_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
            "temperature": temperature,
            "n_new": n_new,
            "reps": reps,
            "prompt_tokens": len(prompt_ids),
            "open_prompt_tokens": len(open_prompt_ids),
        }
    )

    forward_cost = _measure_forward_cost(
        engine, prompt_ids, lengths=sorted({1, 2, 3, max(ks)})
    )
    result["forward_cost"] = forward_cost

    jobs: List[Tuple[str, str, List[int], int]] = []
    for k in ks:
        jobs.append(("repetitive", f"prompt_lookup_k{k}", prompt_ids, k))
    jobs.append(("open_ended", f"prompt_lookup_k{ks[-1]}", open_prompt_ids, ks[-1]))
    # 基线每个 workload 一个
    baseline_jobs = [("repetitive", "baseline_ar", prompt_ids), ("open_ended", "baseline_ar", open_prompt_ids)]

    records: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for rep in range(reps):
        seed = 1234 + 17 * rep
        # 交错：先基线，再各档 k，同一个 rep 内相邻执行
        for prompt_name, name, pids in baseline_jobs:
            r = _run_baseline(engine, pids, n_new, seed=seed, temperature=temperature)
            records.setdefault((prompt_name, name), []).append(r)
        for prompt_name, name, pids, k in jobs:
            r = _run_speculative(
                engine, pids, n_new, k, seed=seed, temperature=temperature
            )
            records.setdefault((prompt_name, name), []).append(r)

    # 每个配置取最快的一次，但保留全部 rep 的耗时
    rows: List[Dict[str, Any]] = []
    for (prompt_name, name), recs in records.items():
        best = min(recs, key=lambda r: r["elapsed_ms"])
        row = dict(best)
        row["prompt"] = prompt_name
        row["reps_ms"] = [round(r["elapsed_ms"], 2) for r in recs]
        row["reps_speedup_spread"] = round(
            max(r["elapsed_ms"] for r in recs) / min(r["elapsed_ms"] for r in recs), 3
        )
        # 只有一次 96-token 的采样，α 的抽样噪声不小，所以同时给 rep 均值
        n = len(recs)
        row["elapsed_ms_mean"] = sum(r["elapsed_ms"] for r in recs) / n
        row["tokens_per_step_mean"] = sum(r["tokens_per_step"] for r in recs) / n
        row["acceptance_rate_mean"] = sum(r["acceptance_rate"] for r in recs) / n
        row["steps_mean"] = sum(r["steps"] for r in recs) / n
        # 每轮真正喂给验证前向的草稿长度 / 每轮发生几次前向
        if name == "baseline_ar":
            row["hit_rate"] = 0.0
            row["draft_len_mean"] = 0.0
            row["draft_len_per_step"] = 0.0
            row["forwards_per_step"] = 1.0
        else:
            hit_steps = max(1.0, row["hit_rate"] * row["steps"])
            row["draft_len_mean"] = row["draft_tokens"] / hit_steps  # 命中草稿时的平均长度
            row["draft_len_per_step"] = row["draft_tokens"] / max(1, row["steps"])
            row["forwards_per_step"] = 1.0 + row["hit_rate"]
        rows.append(row)

    rows.sort(key=lambda r: (r["prompt"] != "repetitive", r["k"]))
    baseline_ms = {r["prompt"]: r["elapsed_ms"] for r in rows if r["name"] == "baseline_ar"}
    baseline_mean = {r["prompt"]: r["elapsed_ms_mean"] for r in rows if r["name"] == "baseline_ar"}
    c1 = forward_cost["single_token_ms"]
    for r in rows:
        if r["name"] == "baseline_ar":
            r["speedup_vs_baseline"] = 1.0
            r["speedup_vs_baseline_mean"] = 1.0
            r["predicted_speedup"] = 1.0
            continue
        r["speedup_vs_baseline"] = baseline_ms[r["prompt"]] / r["elapsed_ms"]
        r["speedup_vs_baseline_mean"] = baseline_mean[r["prompt"]] / r["elapsed_ms_mean"]
        # 每轮成本（以"一次 1-token 前向"为单位）：
        #   1) 把最后一个输出 token 喂回去，恢复下一步 logits —— 恒定 1 次；
        #   2) 只有命中草稿的那些 step 才多付一次验证前向，成本 ≈ ratio(draft_len)。
        # 没有草稿的 step 只付 1)，所以按 k 算会把开放域 prompt 的成本高估一倍。
        d = r["draft_len_mean"]
        ratio = _pass_ratio(forward_cost, d) if d > 0 else 1.0
        r["pass_ratio_at_draft_len"] = ratio
        r["predicted_forwards_per_token"] = (1.0 + r["hit_rate"] * ratio) / r["tokens_per_step"]
        r["predicted_speedup"] = r["tokens_per_step"] / (1.0 + r["hit_rate"] * ratio)
        r["predicted_ms_per_token"] = r["predicted_forwards_per_token"] * c1

    result["runs"] = rows
    result["baseline_ms_per_token"] = {p: (ms / n_new) for p, ms in baseline_ms.items()}
    result["notes"] = (
        "计时只覆盖解码循环（prefill 不计）；temperature=1.0 采样；"
        "每轮成本 = 一次 1-token 前向（把最后一个输出 token 喂回去恢复 next-token logits）"
        " + hit_rate × 一次 draft_len 长的验证前向，所以加速比低于 tokens/step。"
        "predicted_speedup 用隔离实测的前向成本曲线按这个公式算出，"
        "对 GPU 抢占不敏感，可作为交叉验证；reps_ms 是每次 rep 的原始耗时。"
        "每个配置取最快那次计时（基线也取最快，两边对称）。"
    )
    return result


# =========================================================================== #
# 作图
# =========================================================================== #


def make_figure(corr: Dict[str, Any], speed: Dict[str, Any]) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(18.5, 10.5))
    ax_a, ax_b, ax_c, ax_d, ax_e, ax_f = axes.reshape(-1)

    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})

    # ---- (a) 分布对比 ----
    p = corr["config"]["p"]
    q = corr["config"]["q"]
    ok = corr["correct"]["empirical"]
    bad = corr["naive_no_correction"]["empirical"]
    idx = list(range(len(p)))
    w = 0.38
    ax_a.plot(idx, p, "k^-", ms=6, lw=1.2, label="target p")
    ax_a.bar([i - w / 2 - 0.02 for i in idx], ok, width=w, label="correct (verify+residual)",
             color="#1f77b4", alpha=0.85)
    ax_a.bar([i + w / 2 + 0.02 for i in idx], bad, width=w, label="naive (no correction)",
             color="#d62728", alpha=0.85)
    ax_a.set_xticks(idx)
    ax_a.set_xlabel("token id")
    ax_a.set_ylabel("probability")
    ax_a.set_title(
        f"(a) Output distribution, k=1, n={corr['config']['n_trials']:,}\n"
        f"correct L1={corr['correct']['l1']:.4f} vs naive L1={corr['naive_no_correction']['l1']:.4f}"
    )
    ax_a.legend(fontsize=8)

    # ---- (b) L1 距离（对数轴）----
    names = [
        "correct\n(verify+residual)",
        "sampling noise floor\n(two independent halves)",
        "naive: no residual\ncorrection",
        "naive: accept draft\n(no verification)",
    ]
    vals = [
        corr["correct"]["l1"],
        corr["noise_floor"]["l1_between_two_independent_halves"],
        corr["naive_no_correction"]["l1"],
        corr["naive_accept_all"]["l1"],
    ]
    colors = ["#1f77b4", "#7f7f7f", "#d62728", "#8c564b"]
    ax_b.barh(names, vals, color=colors)
    ax_b.set_xscale("log")
    ax_b.set_xlabel("L1 distance to target p (log scale)")
    ax_b.set_title("(b) Correctness: L1 error vs target distribution")
    for i, v in enumerate(vals):
        ax_b.text(v * 1.08, i, f"{v:.4f}", va="center", fontsize=9)
    ax_b.set_xlim(right=max(vals) * 4)

    # ---- (c) speedup vs alpha（k 曲线，cost_ratio=0.1）----
    table = speed["table"]
    cr = 0.1
    rows = [r for r in table if abs(r["cost_ratio"] - cr) < 1e-9]
    ks = sorted({r["k"] for r in rows})
    colors_k = {1: "#1f77b4", 2: "#2ca02c", 3: "#ff7f0e", 5: "#d62728"}
    for k in ks:
        sub = sorted([r for r in rows if r["k"] == k], key=lambda r: r["alpha"])
        a = [r["alpha"] for r in sub]
        ax_c.plot(a, [r["theory_speedup"] for r in sub], "-", color=colors_k[k], label=f"k={k} theory")
        ax_c.plot(
            a,
            [r["mc_speedup"] for r in sub],
            "o",
            ms=4,
            mfc="none",
            color=colors_k[k],
            label=f"k={k} Monte Carlo",
        )
    ax_c.set_xlabel("acceptance rate alpha")
    ax_c.set_ylabel("speedup (x)")
    ax_c.set_title(f"(c) Theory vs Monte Carlo, cost_ratio={cr}\n(100k rounds per point)")
    ax_c.legend(fontsize=7, ncol=2)
    ax_c.axhline(1.0, color="k", lw=0.8, ls=":")

    # ---- (d) speedup vs k：拐点 ----
    curves = speed["curves"]
    shown = [c for c in curves if abs(c["alpha"] - 0.7) < 1e-9]
    colors_cr = {0.05: "#1f77b4", 0.1: "#ff7f0e", 0.3: "#d62728"}
    for c in sorted(shown, key=lambda c: c["cost_ratio"]):
        col = colors_cr.get(round(c["cost_ratio"], 3), None)
        ax_d.plot(c["k"], c["speedup"], "-o", ms=3, color=col, label=f"cost_ratio={c['cost_ratio']}")
        ax_d.plot([c["k_opt"]], [c["peak_speedup"]], "*", ms=16, color=col,
                  markeredgecolor="k", markeredgewidth=0.5)
        ax_d.annotate(
            f"knee k={c['k_opt']}\n{c['peak_speedup']:.2f}x",
            (c["k_opt"], c["peak_speedup"]),
            textcoords="offset points",
            xytext=(8, 6),
            fontsize=8,
        )
    ax_d.set_xlabel("draft length k")
    ax_d.set_ylabel("speedup (x)")
    ax_d.set_title("(d) alpha=0.7: speedup peaks then falls (stars = optimal k)")
    ax_d.legend(fontsize=8, loc="upper right")
    ax_d.axhline(1.0, color="k", lw=0.8, ls=":")
    ax_d.set_ylim(top=max(max(c["speedup"]) for c in shown) * 1.16)

    # ---- (e) prompt-lookup 接受率沿工作负载的走势 ----
    cpu = speed.get("workload_prompt_lookup", {})
    if cpu.get("status") == "ran":
        for k in sorted(cpu["overall"]):
            r = cpu["overall"][k]
            ax_e.plot(r["curve_x"], r["curve_y"], lw=1.4, label=f"{k}: alpha={r['acceptance_rate']:.2f}")
        ax_e.set_xlabel("token position in workload")
        ax_e.set_ylabel("accepted draft tokens (rolling mean, window=50)")
        ax_e.set_title(
            f"(e) Prompt-lookup on {cpu['workload']['total_tokens']} tok synthetic workload\n"
            "(oracle acceptance: ground truth = the text itself)"
        )
        ax_e.legend(fontsize=8)
    else:
        ax_e.text(0.5, 0.5, "not run", ha="center", va="center", transform=ax_e.transAxes)
        ax_e.set_title("(e) prompt-lookup acceptance")

    # ---- (f) 真实模型：tokens/step 与端到端加速比 ----
    gpu = speed.get("real_model", {})
    if gpu.get("status") == "ran":
        runs = [r for r in gpu["runs"]]
        labels = [f"{r['prompt']}\n{r['name']}" for r in runs]
        tps = [r["tokens_per_step_mean"] for r in runs]
        spd = [r["speedup_vs_baseline_mean"] for r in runs]
        xpos = list(range(len(runs)))
        ax_f.bar([x - 0.2 for x in xpos], tps, width=0.4, color="#1f77b4", label="tokens / step")
        ax_f.set_ylabel("tokens per step", color="#1f77b4")
        ax_f.set_xticks(xpos)
        ax_f.set_xticklabels(labels, fontsize=7, rotation=20, ha="right")
        ax2 = ax_f.twinx()
        ax2.bar([x + 0.2 for x in xpos], spd, width=0.4, color="#d62728", label="end-to-end speedup")
        ax2.axhline(1.0, color="k", lw=0.8, ls=":")
        ax2.set_ylabel("speedup vs baseline (x)", color="#d62728")
        ax_f.set_title(
            f"(f) Real Qwen2.5-1.5B fp16, RTX 4050 6GB\n"
            f"1-token fwd {gpu['forward_cost']['single_token_ms']:.1f} ms, "
            f"5-token fwd {gpu['forward_cost']['k_token_pass_ms']:.1f} ms"
        )
        h1, l1 = ax_f.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax_f.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")
    else:
        reason = gpu.get("reason", "unknown") or "unknown"
        # 中文在默认字体下会变成方框，图上一律用 ASCII
        reason = reason.encode("ascii", "replace").decode("ascii")[:180]
        ax_f.text(
            0.5,
            0.5,
            "real model: not run\n" + reason[:180],
            ha="center",
            va="center",
            wrap=True,
            transform=ax_f.transAxes,
            fontsize=9,
        )
        ax_f.set_title("(f) real model experiment")

    fig.suptitle(
        "Speculative decoding: distribution-preserving verification, speedup model, real workload",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(FIG_PATH, dpi=150)
    plt.close(fig)
    return FIG_PATH


# =========================================================================== #
# main
# =========================================================================== #


def _build_open_prompt(tokenizer) -> List[int]:
    text = (
        "请用自己的话解释：为什么投机解码不会改变大模型的输出分布？"
        "如果验证时忘了做残差修正，会发生什么？请举一个具体的例子。"
    )
    return tokenizer(text)["input_ids"]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=200_000)
    ap.add_argument("--n-scalar", type=int, default=20_000)
    ap.add_argument("--rounds", type=int, default=100_000)
    ap.add_argument("--n-new", type=int, default=96)
    ap.add_argument("--reps", type=int, default=5, help="C2 每个配置重复次数（交错执行，取最快）")
    ap.add_argument("--skip-model", action="store_true", help="跳过 C2（真实模型）")
    args = ap.parse_args(argv)

    set_seed(0)
    print("=" * 78)
    print("spec_decode benchmark")
    print("env:", json.dumps(device_info(), ensure_ascii=False))
    print("=" * 78)

    # ---- 树的正确性测试（必须先过） ----
    print("\n[0] tree attention 正确性测试")
    test_tree_mask(verbose=True)

    # ---- A ----
    print("\n[A] 分布等价性")
    corr = experiment_a(n_trials=args.n_trials, n_scalar=args.n_scalar)
    corr["tree_mask_test"] = {"status": "PASS"}
    rows = [
        {
            "method": corr["correct"]["name"],
            "L1": corr["correct"]["l1"],
            "max_abs_err": corr["correct"]["max_abs_err"],
            "chi2": corr["correct"].get("chi2"),
        },
        {
            "method": corr["naive_no_correction"]["name"],
            "L1": corr["naive_no_correction"]["l1"],
            "max_abs_err": corr["naive_no_correction"]["max_abs_err"],
            "chi2": corr["naive_no_correction"].get("chi2"),
        },
        {
            "method": corr["naive_accept_all"]["name"],
            "L1": corr["naive_accept_all"]["l1"],
            "max_abs_err": corr["naive_accept_all"]["max_abs_err"],
            "chi2": corr["naive_accept_all"].get("chi2"),
        },
    ]
    print_table(rows)
    print(
        f"  noise floor (two halves L1) = {corr['noise_floor']['l1_between_two_independent_halves']:.4f}\n"
        f"  acceptance rate: theory 1-TV = {corr['theory_acceptance_rate']:.4f}, "
        f"measured = {corr['measured_acceptance_rate']:.4f}\n"
        f"  scalar-vs-batch: L1 = {corr['scalar_vs_batch']['l1_scalar_vs_batch_empirical']:.4f}, "
        f"scalar L1 to p = {corr['scalar_vs_batch']['l1_scalar_vs_p']:.4f} "
        f"({corr['scalar_vs_batch']['n_scalar']} samples)"
    )
    p_corr = save_result("spec_correctness", corr)
    print("  saved:", p_corr)

    # ---- B ----
    print("\n[B] alpha vs speedup（理论 vs 蒙特卡洛）")
    speed = experiment_b(rounds=args.rounds)
    sub = [r for r in speed["table"] if abs(r["cost_ratio"] - 0.1) < 1e-9]
    print_table(
        [
            {
                "alpha": r["alpha"],
                "k": r["k"],
                "theory": round(r["theory_speedup"], 4),
                "mc": round(r["mc_speedup"], 4),
                "rel_err": round(r["rel_err"], 5),
            }
            for r in sub[:12]
        ]
    )
    print(f"  max rel err (speedup) = {speed['max_rel_err_speedup']:.5f}, "
          f"max rel err (tokens/step) = {speed['max_rel_err_tokens_per_step']:.5f}")
    print_table(
        [
            {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
            for r in speed["optimal_k"]["rows"][:9]
        ],
        columns=["alpha", "cost_ratio", "k_opt", "peak_speedup", "speedup_at_k1"],
    )
    ex = speed["knee_summary"]["example"]
    if ex:
        print(f"  knee example: alpha={ex['alpha']} cost_ratio={ex['cost_ratio']} "
              f"k_opt={ex['k_opt']} peak={ex['peak_speedup']:.3f}x, "
              f"at k={speed['knee_summary']['k_max'] if 'k_max' in speed['knee_summary'] else speed['optimal_k']['k_max']} "
              f"only {ex['speedup_at_kmax']:.3f}x")
    speed["tree_mask_test"] = {"status": "PASS"}
    speed["real_model"] = {"status": "not_run", "reason": "pending"}
    p_speed = save_result("spec_speedup", speed)
    print("  saved:", p_speed)

    # ---- C1 ----
    print("\n[C1] prompt-lookup 在真实文本上的接受率（CPU, 无模型）")
    tok = None
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(_resolve_model_dir(MODEL_CACHE))
    except Exception as exc:  # tokenizer 不可用就退化成字节序列
        print("  tokenizer 不可用，退化到字节级 pseudo-token:", type(exc).__name__)

    if tok is not None:
        text, ids, spans = _build_workload(tok, {})
    else:
        text, ids, spans = _build_workload(_ByteTokenizer(), {})
    print(f"  workload: {len(ids)} tokens, segments: {[(s['name'], s['tokens']) for s in spans]}")
    cpu_res = experiment_c_cpu(ids, spans, ks=(3, 5))
    for k, r in cpu_res["overall"].items():
        print(f"  {k}: hit_rate={r['hit_rate']:.3f} acceptance={r['acceptance_rate']:.3f} "
              f"tokens/step={r['tokens_per_step']:.3f} drafted={r['draft_tokens']}")
    print_table(
        [
            {"segment": s["segment"], "tokens": s["tokens"],
             "hit_rate": round(s["hit_rate"], 3),
             "acceptance": round(s["acceptance_rate"], 3),
             "tokens/step": round(s["tokens_per_step"], 3)}
            for s in cpu_res["by_segment"]
        ]
    )
    speed["workload_prompt_lookup"] = cpu_res

    # ---- C2 ----
    print("\n[C2] 真实模型（Qwen2.5-1.5B-Instruct, fp16）")
    if args.skip_model:
        speed["real_model"] = {"status": "not_run", "reason": "skipped by --skip-model"}
        print("  skipped")
    else:
        try:
            log_span = next(s for s in spans if s["name"] == "structured_log")
            prompt_ids = ids[log_span["start"]: log_span["start"] + 200]
            open_ids = _build_open_prompt(tok) if tok is not None else ids[:20]
            gpu_res = experiment_c_gpu(
                prompt_ids, open_ids, n_new=args.n_new, reps=args.reps
            )
            print_table(
                [
                    {
                        "prompt": r["prompt"],
                        "cfg": r["name"],
                        "k": r["k"],
                        "tokens/step": round(r["tokens_per_step"], 3),
                        "tps_mean": round(r["tokens_per_step_mean"], 3),
                        "accept_mean": round(r["acceptance_rate_mean"], 3),
                        "ms": round(r["elapsed_ms"], 1),
                        "speedup": round(r["speedup_vs_baseline"], 3),
                        "spd_mean": round(r["speedup_vs_baseline_mean"], 3),
                        "predicted": round(r["predicted_speedup"], 3),
                    }
                    for r in gpu_res.get("runs", [])
                ]
            )
            if gpu_res.get("status") == "ran":
                print("  forward cost:", json.dumps(gpu_res["forward_cost"], ensure_ascii=False))
                print("  vram peak GB:", gpu_res.get("vram_peak_gb"))
            speed["real_model"] = gpu_res
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            print("  FAILED, recorded as not_run:", msg)
            traceback.print_exc()
            speed["real_model"] = {"status": "not_run", "reason": msg[:500]}

    p_speed = save_result("spec_speedup", speed)
    print("  saved:", p_speed)

    # ---- 图 ----
    fig = make_figure(corr, speed)
    print("\nfigure:", fig)

    print("\nDONE")
    return 0


class _ByteTokenizer:
    """tokenizer 不可用时的兜底：UTF-8 字节当 token（仅用于 C1 的相对趋势）。"""

    def __call__(self, text: str) -> Dict[str, List[int]]:
        return {"input_ids": list(text.encode("utf-8"))}


if __name__ == "__main__":
    raise SystemExit(main())