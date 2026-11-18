"""分页 KV cache 上的 decode attention。

decode 阶段的 attention 是「1 个 query token 对 n 个 key」，
和 prefill 的 N×N 完全不是一个形状——这也决定了它的瓶颈：
  - prefill: 计算密集，靠 tensor core 的 tl.dot 打满
  - decode : **访存密集**，每个 token 只做 2nD 次乘加，却要把 n 个 K/V 全读一遍
所以 decode kernel 的优化目标是"少读、合并读"，而不是"多做 FMA"。

这里给两条路径，并且**故意都保留**，因为它们回答的是同一个问题的两面：
  paged_attention_gather : 先按 block_table 把 K/V gather 成连续，再调 SDPA。
                           简单、能复用高度优化的 GEMM，但多一次 n×D 的拷贝
                           —— 等于把分页省下的搬运算又搬了一遍。
  paged_attention_fused  : Triton kernel 直接按 block_table 读物理块，
                           零拷贝，但没有 tl.dot（decode 的 M=1 用不上 tensor core），
                           靠逐元素乘加 + warp reduce 累加。

顺带一个常被忽略的事实：PagedAttention 的 kernel 本身**比连续布局更慢**。
vLLM 论文 §7.1 明确写了 "this leads to 20–26% higher attention kernel latency,
compared to the highly-optimized FasterTransformer implementation"。
它仍然值得，是因为省下的显存让 batch 变大，总吞吐反而上去了
——这个 trade-off 在报告里单独一段说明，不然会被问"你为什么用更慢的 kernel"。
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

LOG2E = tl.constexpr(1.4426950408889634)
_NEG = tl.constexpr(-1.0e6)


@triton.jit
def _paged_decode_fwd(
    Q,
    KC,
    VC,
    BT,
    ContextLens,
    Out,
    sm_scale,
    stride_qb,
    stride_qh,
    stride_kb,
    stride_kbs,
    stride_kh,
    stride_ob,
    stride_oh,
    stride_btb,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """一个 program 负责一个 (batch, head) 的单 token decode。

    KCache/VCache 布局: [num_blocks, BLOCK_N, H, D]
    BT 布局:            [B, max_blocks] int32
    Q/Out 布局:         [B, H, D]
    """
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H

    ctx = tl.load(ContextLens + b)
    offs_d = tl.arange(0, D)
    q = tl.load(Q + b * stride_qb + h * stride_qh + offs_d).to(tl.float32)

    acc = tl.zeros([D], dtype=tl.float32)
    m_i = tl.full([], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([], dtype=tl.float32)

    n_blocks = (ctx + BLOCK_N - 1) // BLOCK_N
    offs_n = tl.arange(0, BLOCK_N)

    for bi in range(0, n_blocks):
        bid = tl.load(BT + b * stride_btb + bi)
        pos = bi * BLOCK_N + offs_n
        mask = pos < ctx

        k_ptr = KC + bid.to(tl.int64) * stride_kb + offs_n[:, None] * stride_kbs \
            + h * stride_kh + offs_d[None, :]
        k = tl.load(k_ptr, mask=mask[:, None], other=0.0).to(tl.float32)

        qk = tl.sum(k * q[None, :], 1) * sm_scale
        qk = tl.where(mask, qk, _NEG)

        m_new = tl.maximum(m_i, tl.max(qk, 0))
        p = tl.math.exp2((qk - m_new) * LOG2E)
        alpha = tl.math.exp2((m_i - m_new) * LOG2E)

        v_ptr = VC + bid.to(tl.int64) * stride_kb + offs_n[:, None] * stride_kbs \
            + h * stride_kh + offs_d[None, :]
        v = tl.load(v_ptr, mask=mask[:, None], other=0.0).to(tl.float32)

        acc = acc * alpha + tl.sum(p[:, None] * v, 0)
        l_i = l_i * alpha + tl.sum(p, 0)
        m_i = m_new

    out = acc / l_i
    tl.store(Out + b * stride_ob + h * stride_oh + offs_d, out.to(Out.dtype.element_ty))


def paged_attention_fused(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    context_lens: torch.Tensor,
) -> torch.Tensor:
    """q: (B, H, D) 单 token；k_cache/v_cache: (num_blocks, block_size, H, D)。"""
    B, H, D = q.shape
    assert q.is_cuda and k_cache.is_cuda
    block_size = k_cache.shape[1]
    out = torch.empty_like(q)
    # stride_kh 是 (block_size, H, D) 里 H 维的跨度
    stride_kb, stride_kbs, stride_kh, _ = k_cache.stride()
    stride_vb, stride_vbs, stride_vh, _ = v_cache.stride()
    assert stride_kh == stride_vh, "K/V 的 head stride 必须一致才能共用一套索引"

    grid = (B * H,)
    _paged_decode_fwd[grid](
        q, k_cache, v_cache, block_table, context_lens, out,
        1.0 / math.sqrt(D),
        q.stride(0), q.stride(1),
        stride_kb, stride_kbs, stride_kh,
        out.stride(0), out.stride(1),
        block_table.stride(0),
        H=H,
        D=D,
        BLOCK_N=block_size,
    )
    return out


def paged_attention_gather(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    context_lens: torch.Tensor,
) -> torch.Tensor:
    """朴素路径：先把 K/V 按 block_table 搬成连续布局，再交给 torch SDPA。

    这里**必须**有两条路径，因为基准测试的公平性全押在这上面：
      - 慢路径（逐请求 Python 循环）：只在 block_table 布局不规整时用。
        实测它比批量化慢 20–200 倍，而且慢的是 Python/launch 开销，
        不是 attention 本身。拿它当基线会得出"我的 kernel 快 200 倍"这种假结论。
      - 快路径（批量 gather + 一次 SDPA）：所有请求上下文等长时可用，
        这才是 vLLM 关闭分页 kernel 时的对照物。

    布局要点（我在这里错过一次，值得写下来）：
      k_cache[ids] 的形状是 (nb, block_size, H, D)，**直接** reshape(-1, H, D)
      就是正确的 token 顺序；再加 permute 会把 H 夹到中间，静默算错。
    """
    B, H, D = q.shape
    block_size = k_cache.shape[1]
    lens = context_lens.tolist()
    uniform = len(set(lens)) == 1

    if uniform and k_cache.is_contiguous() and v_cache.is_contiguous():
        n = int(lens[0])
        nb = (n + block_size - 1) // block_size
        ids = block_table[:, :nb].long()                                  # (B, nb)
        offs = torch.arange(block_size, device=q.device)
        flat = (ids.unsqueeze(-1) * block_size + offs).reshape(B, nb * block_size)[:, :n]
        k = k_cache.reshape(-1, H, D)[flat].transpose(1, 2)               # (B, H, n, D)
        v = v_cache.reshape(-1, H, D)[flat].transpose(1, 2)
        o = torch.nn.functional.scaled_dot_product_attention(q.unsqueeze(2), k, v)
        return o.squeeze(2)

    outs = []
    for b in range(B):
        n = int(context_lens[b])
        nb = (n + block_size - 1) // block_size
        ids = block_table[b, :nb].long()
        k = k_cache[ids].reshape(-1, H, D)[:n]   # (n, H, D)，无需 permute
        v = v_cache[ids].reshape(-1, H, D)[:n]
        o = torch.nn.functional.scaled_dot_product_attention(
            q[b].unsqueeze(0).unsqueeze(2),  # (1, H, 1, D)
            k.transpose(0, 1).unsqueeze(0),  # (1, H, n, D)
            v.transpose(0, 1).unsqueeze(0),
        )
        outs.append(o.squeeze(0).squeeze(1))
    return torch.stack(outs, dim=0)


def paged_attention_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    context_lens: torch.Tensor,
) -> torch.Tensor:
    """纯 fp32 逐元素参考实现，用来验证上面两条路径。"""
    B, H, D = q.shape
    block_size = k_cache.shape[1]
    scale = 1.0 / math.sqrt(D)
    out = torch.empty(B, H, D, device=q.device, dtype=torch.float32)
    for b in range(B):
        n = int(context_lens[b])
        nb = (n + block_size - 1) // block_size
        ids = block_table[b, :nb].long()
        k = k_cache[ids].reshape(-1, H, D)[:n].float()
        v = v_cache[ids].reshape(-1, H, D)[:n].float()
        for h in range(H):
            qk = (k[:, h, :] * q[b, h].float()[None, :]).sum(-1) * scale
            p = torch.softmax(qk, dim=-1)
            out[b, h] = (p[:, None] * v[:, h, :]).sum(0)
    return out