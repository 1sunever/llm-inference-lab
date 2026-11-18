"""PagedAttention 的四组实验。

注意分工：A/B 两组是**纯记账仿真**（不碰 GPU），因为要跑上千条请求；
C 组是正确性；D 组才是 kernel 延迟。混在一起测会既慢又说不清。

A. 碎片：预分配 vs 分页，同样预算能装多少条序列
B. 前缀命中率：多轮对话里 system prompt 的复用，以及 LRU 驱逐的代价
C. 正确性：四条路径（连续布局 / paged gather / paged fused / fp32 参考）互相验证
D. decode 延迟：fused kernel vs gather+SDPA，看"零拷贝"到底省了多少

运行：
    /home/sunever/aiinfra/.venv/bin/python src/llm_lab/paged_attn/bench.py
"""

from __future__ import annotations

import math
import os
import random
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import torch

from llm_lab.common import bench, print_table, save_result

warnings.filterwarnings("ignore", category=UserWarning, module="triton")

from llm_lab.paged_attn.block_manager import BlockManager  # noqa: E402
from llm_lab.paged_attn.cache import ContiguousKVCache, PagedKVCache  # noqa: E402
from llm_lab.paged_attn.paged_decode import (  # noqa: E402
    paged_attention_fused,
    paged_attention_gather,
    paged_attention_reference,
)
from llm_lab.paged_attn.prefix_cache import PrefixCache  # noqa: E402

BLOCK_SIZE = 16

# Qwen2.5-1.5B 的 KV 账：28 层、2 个 KV 头（GQA）、head_dim 128、fp16
# 每 token = 2(K和V) × 2(kv头) × 128 × 2字节 × 28层 = 28 KB
KV_BYTES_PER_TOKEN = 2 * 2 * 128 * 2 * 28


# ------------------------------------------------------------------ A


def exp_fragmentation() -> dict:
    """同样显存预算下，两种方案各能服务多少条序列。"""
    rng = random.Random(0)
    n_seqs = 256
    # 真实请求长度是长尾的：多数几百 token，少数几千
    lengths = [int(min(4096, max(32, rng.lognormvariate(math.log(600), 0.8)))) for _ in range(n_seqs)]
    max_len = 4096  # 预分配方案必须按上界要空间

    # --- 连续预分配 ---
    contig_alloc = n_seqs * max_len
    contig_used = sum(lengths)
    contig_waste = 1.0 - contig_used / contig_alloc

    # --- 分页 ---
    blocks = [math.ceil(n / BLOCK_SIZE) for n in lengths]
    paged_alloc = sum(blocks) * BLOCK_SIZE
    paged_waste = 1.0 - contig_used / paged_alloc

    # --- 同样预算能装多少条"平均长度"的序列 ---
    budget_bytes = 2.9 * 1024**3  # 6GB 卡扣掉 1.5B 权重后大约能留给 KV 的量
    budget_tokens = budget_bytes / KV_BYTES_PER_TOKEN
    mean_len = sum(lengths) / len(lengths)

    # 预分配：每条都要 max_len 的空间
    seqs_contig = budget_tokens / max_len
    # 分页：每条平均占用 ceil(mean_len/bs)*bs，碎片只有半块
    seqs_paged = budget_tokens / (math.ceil(mean_len / BLOCK_SIZE) * BLOCK_SIZE)

    rows = [
        {"scheme": "contiguous(prealloc)", "allocated_tokens": contig_alloc,
         "used_tokens": contig_used, "waste_pct": contig_waste * 100,
         "concurrent_seqs": round(seqs_contig, 1)},
        {"scheme": f"paged(bs={BLOCK_SIZE})", "allocated_tokens": paged_alloc,
         "used_tokens": contig_used, "waste_pct": paged_waste * 100,
         "concurrent_seqs": round(seqs_paged, 1)},
    ]
    print("\n=== A. 碎片与并发能力 ===")
    print(f"序列数 {n_seqs}，长度 lognormal(mean≈600)，max_len={max_len}，"
          f"KV {KV_BYTES_PER_TOKEN/1024:.1f} KB/token，预算 {budget_tokens/1e6:.2f} M token")
    print_table(rows)

    return {
        "config": {"n_seqs": n_seqs, "max_len": max_len, "block_size": BLOCK_SIZE,
                   "kv_bytes_per_token": KV_BYTES_PER_TOKEN,
                   "budget_tokens": budget_tokens, "mean_len": mean_len},
        "lengths": lengths,
        "rows": rows,
        "speedup_concurrency": seqs_paged / seqs_contig,
    }


# ------------------------------------------------------------------ B


def _allocate_keep(pc: PrefixCache, bm: BlockManager, token_ids: list[int]):
    """为一次请求分配并**保持**占用（模拟它正在 decode），返回
    (命中块数, 需要块数, block_table, 被挤掉的块数)。

    必须保持占用才能制造显存压力：如果每个请求中途就释放，
    池子永远够用，命中率会变成一个假的 85%（这个坑我踩过）。
    """
    needed = math.ceil(len(token_ids) / BLOCK_SIZE)
    hit = pc.get_computed_blocks(token_ids)
    n_hit = len(hit)
    extra = needed - n_hit
    clamped = 0
    if extra > 0:
        # 不够就按 LRU 驱逐，最多试几轮，避免死循环
        for _ in range(8):
            if bm.num_free_blocks >= extra:
                break
            if pc.evict(extra) == 0:
                break
        got = min(extra, bm.num_free_blocks)
        clamped = extra - got
        new = pc.alloc_blocks(got) if got > 0 else []
        hit = hit + new
    pc.register(hit, token_ids)
    return n_hit, needed, hit, clamped


def exp_prefix_cache() -> dict:
    """多轮对话：固定 1024 token 的 system prompt + 逐轮增长的历史。

    并发 4 个会话同时 decode，这样 128 块的池子会真的不够用，
    才能看出「LRU 驱逐会吃掉命中率」这件事。
    """
    system_prompt = list(range(1000, 1000 + 1024))
    rng = random.Random(1)
    n_sessions, turns = 4, 4

    # 先构造每条会话每一轮完整的输入 token
    per_session: list[list[list[int]]] = []
    for _ in range(n_sessions):
        history: list[int] = []
        reqs = []
        for _ in range(turns):
            user = [rng.randrange(100_000, 200_000) for _ in range(96)]
            reqs.append(system_prompt + history + user)
            history = history + user + [rng.randrange(200_000, 300_000) for _ in range(120)]
        per_session.append(reqs)

    pools = {"large_pool(4096 blocks)": 4096,
             "tight_pool(512 blocks)": 512,
             "very_tight(128 blocks)": 128}

    rows = []
    for name, num_blocks in pools.items():
        bm = BlockManager(num_blocks, BLOCK_SIZE)
        pc = PrefixCache(bm)
        hit_total = need_total = clamp_total = 0
        for t in range(turns):
            tables = []
            # 同一轮里所有会话并发占用
            for s in range(n_sessions):
                h, need, table, clamped = _allocate_keep(pc, bm, per_session[s][t])
                hit_total += h
                need_total += need
                clamp_total += clamped
                tables.append(table)
            for tb in tables:
                bm.free(tb)
        rows.append({
            "workload": name,
            "pool_blocks": num_blocks,
            "hit_blocks": hit_total,
            "query_blocks": need_total,
            "hit_rate_pct": 100.0 * hit_total / need_total if need_total else 0.0,
            "prefill_tokens_saved": hit_total * BLOCK_SIZE,
            "total_tokens": need_total * BLOCK_SIZE,
            # >0 说明池子小到连请求都装不下，命中率这个指标已经不可信
            "clamped_blocks": clamp_total,
        })

    print("\n=== B. 前缀命中率（4 会话并发 × 4 轮）===")
    print_table(rows)
    return {"config": {"block_size": BLOCK_SIZE, "system_prompt_tokens": 1024,
                       "concurrent_sessions": n_sessions, "turns": turns},
            "rows": rows}


# ------------------------------------------------------------------ C


def exp_correctness() -> dict:
    """四条路径互相验证。这是唯一能证明"分页没算错"的东西。"""
    torch.manual_seed(0)
    B, H, D = 4, 4, 64
    lengths = [16, 33, 64, 100]  # 故意含非整数倍长度
    max_len = max(lengths)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    k_all = torch.randn(B, max_len, H, D, device=dev, dtype=torch.float16)
    v_all = torch.randn(B, max_len, H, D, device=dev, dtype=torch.float16)

    nb = (max_len + BLOCK_SIZE - 1) // BLOCK_SIZE * B
    paged = PagedKVCache(nb, BLOCK_SIZE, H, D, dtype=torch.float16, device=dev)
    contig = ContiguousKVCache(B, max_len, H, D, dtype=torch.float16, device=dev)

    ctx = []
    for b in range(B):
        n = lengths[b]
        paged.new_sequence(b)
        paged.append(b, k_all[b, :n], v_all[b, :n])
        contig.append(b, k_all[b, :n], v_all[b, :n])
        ctx.append(n)
    ctx_t = torch.tensor(ctx, device=dev, dtype=torch.int32)

    q = torch.randn(B, H, D, device=dev, dtype=torch.float16)
    max_bt = math.ceil(max_len / BLOCK_SIZE)
    bt = torch.zeros(B, max_bt, device=dev, dtype=torch.int32)
    for b in range(B):
        tbl = paged.block_table(b)
        bt[b, : tbl.numel()] = tbl

    # _kv 是 [num_blocks, 2, block_size, H, D]，切出 K/V 两个视图
    # （它们是非连续的，正好顺带验证 kernel 用 stride 而不是假设连续）
    kc, vc = paged._kv[:, 0], paged._kv[:, 1]

    ref = paged_attention_reference(q, kc, vc, bt, ctx_t)
    fused = paged_attention_fused(q, kc, vc, bt, ctx_t)
    gathered = paged_attention_gather(q, kc, vc, bt, ctx_t)

    # 连续布局对照：直接 SDPA
    contig_out = []
    for b in range(B):
        k, v = contig.read(b)
        o = torch.nn.functional.scaled_dot_product_attention(
            q[b].unsqueeze(0).unsqueeze(2),      # (1, H, 1, D)
            k.transpose(0, 1).unsqueeze(0),      # (1, H, n, D)
            v.transpose(0, 1).unsqueeze(0),
        )
        contig_out.append(o.squeeze(0).squeeze(1))
    contig_out = torch.stack(contig_out)

    def err(a, b):
        return (a.float() - b.float()).abs().max().item()

    res = {
        "fused_vs_ref": err(fused, ref),
        "gather_vs_ref": err(gathered, ref),
        "contig_vs_ref": err(contig_out, ref),
        "lengths": lengths,
        "block_size": BLOCK_SIZE,
    }
    print("\n=== C. 正确性（四条路径 vs fp32 参考）===")
    print(f"  fused    max|err| = {res['fused_vs_ref']:.5f}")
    print(f"  gather   max|err| = {res['gather_vs_ref']:.5f}")
    print(f"  contig   max|err| = {res['contig_vs_ref']:.5f}")
    res["pass"] = max(res["fused_vs_ref"], res["gather_vs_ref"], res["contig_vs_ref"]) < 2e-2
    print(f"  verdict: {'PASS' if res['pass'] else 'FAIL'}（fp16 容差 2e-2）")

    del paged, contig
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return res


# ------------------------------------------------------------------ D


def exp_decode_latency() -> dict:
    """fused 分页 kernel vs gather+SDPA。"""
    if not torch.cuda.is_available():
        return {"status": "skipped", "reason": "no cuda"}

    B, H, D = 8, 8, 64
    rows = []
    for ctx in (128, 256, 512, 1024, 2048):
        nb = (ctx + BLOCK_SIZE - 1) // BLOCK_SIZE
        total_blocks = nb * B
        k_cache = torch.randn(total_blocks, BLOCK_SIZE, H, D, device="cuda", dtype=torch.float16)
        v_cache = torch.randn(total_blocks, BLOCK_SIZE, H, D, device="cuda", dtype=torch.float16)
        bt = torch.arange(total_blocks, device="cuda", dtype=torch.int32).reshape(B, nb)
        ctx_t = torch.full((B,), ctx, device="cuda", dtype=torch.int32)
        q = torch.randn(B, H, D, device="cuda", dtype=torch.float16)

        t_f = bench(lambda: paged_attention_fused(q, k_cache, v_cache, bt, ctx_t),
                    warmup=10, rep=50)
        t_g = bench(lambda: paged_attention_gather(q, k_cache, v_cache, bt, ctx_t),
                    warmup=10, rep=50)

        rows.append({
            "ctx_len": ctx, "batch": B, "heads": H, "head_dim": D,
            "fused_ms": t_f["median_ms"], "gather_ms": t_g["median_ms"],
            "speedup_fused_vs_gather": t_g["median_ms"] / t_f["median_ms"],
        })
        print(f"  ctx={ctx:<5} fused={t_f['median_ms']:.4f}ms  "
              f"gather={t_g['median_ms']:.4f}ms  x{rows[-1]['speedup_fused_vs_gather']:.2f}")

        del k_cache, v_cache, q
        torch.cuda.empty_cache()

    print("\n=== D. decode 延迟 ===")
    print_table(rows, ["ctx_len", "fused_ms", "gather_ms", "speedup_fused_vs_gather"])
    return {"config": {"batch": B, "heads": H, "head_dim": D, "block_size": BLOCK_SIZE},
            "rows": rows}


# ------------------------------------------------------------------ 图


def _plot(frag, prefix, decode) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from llm_lab.common import FIGURES_DIR

    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    # (1) 碎片
    ax = axes[0]
    names = ["contiguous\n(prealloc)", f"paged\n(bs={BLOCK_SIZE})"]
    alloc = [r["allocated_tokens"] / 1e6 for r in frag["rows"]]
    used = [r["used_tokens"] / 1e6 for r in frag["rows"]]
    x = range(len(names))
    ax.bar([i - 0.2 for i in x], alloc, width=0.4, label="allocated", color="#d9534f")
    ax.bar([i + 0.2 for i in x], used, width=0.4, label="actually used", color="#5cb85c")
    for i, r in enumerate(frag["rows"]):
        ax.text(i, max(alloc[i], used[i]) * 1.02, f"waste {r['waste_pct']:.1f}%",
                ha="center", fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("tokens (millions)")
    ax.set_title(f"Fragmentation ({frag['config']['n_seqs']} seqs)")
    ax.legend(fontsize=8)

    # (2) 前缀命中率
    ax = axes[1]
    labels = [r["workload"] for r in prefix["rows"]]
    rates = [r["hit_rate_pct"] for r in prefix["rows"]]
    bars = ax.bar(range(len(labels)), rates, color="#337ab7")
    for i, r in enumerate(prefix["rows"]):
        ax.text(i, r["hit_rate_pct"] + 1, f"{r['hit_rate_pct']:.1f}%", ha="center", fontsize=8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([l.replace("(", "\n(") for l in labels], fontsize=7)
    ax.set_ylabel("prefix hit rate (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Prefix reuse under memory pressure")
    ax.axhline(0, color="k", lw=0.5)

    # (3) decode 延迟
    ax = axes[2]
    if decode.get("rows"):
        rs = decode["rows"]
        ax.plot([r["ctx_len"] for r in rs], [r["fused_ms"] for r in rs], "-o", label="fused paged")
        ax.plot([r["ctx_len"] for r in rs], [r["gather_ms"] for r in rs], "--s", label="gather+SDPA")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("context length")
        ax.set_ylabel("latency (ms)")
        ax.set_title("Decode attention (B=8,H=8,D=64)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "no cuda", ha="center")

    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "paged_attn.png")
    fig.savefig(out, dpi=150)
    print(f"\n图已保存 {out}")


def main() -> None:
    frag = exp_fragmentation()
    save_result("paged_fragmentation", frag)

    prefix = exp_prefix_cache()
    save_result("paged_prefix", prefix)

    corr = exp_correctness()
    save_result("paged_correctness", corr)

    decode = exp_decode_latency()
    save_result("paged_decode", decode)

    _plot(frag, prefix, decode)


if __name__ == "__main__":
    main()