"""FlashAttention-2 的 Triton 实现（前向）。

为什么自己写而不是 pip install flash-attn：
  1. 官方包要 nvcc 现场编译，6GB 笔记本上编译动辄 1–2 小时，出问题极难定位；
     Triton 版几秒就 JIT 完，改一个 BLOCK 重跑只需一次 kernel 编译。
  2. 面试要问的是"你知不知道它为什么快"：能改 BLOCK_M/BLOCK_N、能指着
     profile 说"这里 register spill 了"，比装一个包有说服力。
  3. 推理场景 99% 的 attention 是前向（decode 逐 token），前向才是瓶颈。

算法要点（对照 Dao 2023, arXiv:2307.08691）：
  标准 attention 的中间矩阵 S = QK^T 是 N×N，写回 HBM 再读出来做 softmax，
  显存 O(N²)、访存 O(N²)。FlashAttention 把序列切成 BLOCK_M × BLOCK_N 瓦片，
  在 SRAM 里做 online softmax（维护 running max m 与 running sum l），
  任何时刻只持有 BLOCK_M × HEAD_DIM 的累加器，中间量从不落 HBM。
  → 显存 O(N²) → O(N)；HBM 读写 O(N²) → O(N²/M)。快在"少搬数据"，不在"少算"。

数值细节：用 exp2 代替 exp。exp(x) = exp2(x·log2 e)，把乘 log2 e 折进 qk_scale，
省掉一次逐元素乘法；N 大且 BLOCK 小时这项是可测的。

实现约束（写在这里，因为它是正确性的一部分）：
  - 要求 (B,H,N,D) contiguous，stride 由 wrapper 显式传入；
  - 要求 head_dim ∈ {16,32,64,128,256}（见 _CONFIGS）；
  - **N 不要求是 BLOCK 的整数倍**：K/V 的 block_ptr 带 boundary_check，
    非匀速的尾部块用 EVEN_N=False 分支显式把越界列 mask 成 -inf；
  - 只实现前向。反向需要重算 S（recomputation）与 dQ/dK/dV 的原子累加，
    不在本项目范围内，README 的"局限"一节里明确写了这一点。
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

# 注意 Triton >= 3.7 不允许 kernel 里直接读普通 Python 全局变量，
# 必须显式包成 tl.constexpr，否则编译期报 NameError。
LOG2E = tl.constexpr(1.4426950408889634)  # log2(e)
_NEG = tl.constexpr(-1.0e6)  # softmax 里代表 -inf 的有限值（有限值避免 nan 传播）


@triton.jit
def _attn_fwd_inner(
    acc,
    l_i,
    m_i,
    q,
    K_block_ptr,
    V_block_ptr,
    start_m,
    qk_scale,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
    offs_m,
    offs_n,
    N_CTX,
    EVEN_N: tl.constexpr,
):
    """沿 K/V 的序列维扫一遍，累积 online softmax。

    STAGE 用三个取值区分"这段是否需要掩码"，这是 FA2 相对 FA1 的主要收益：
      STAGE=1  纯因果区（整块都在对角线以下）：不需要任何 mask，走快路径
      STAGE=2  对角块：需要三角 mask
      STAGE=3  非因果（全序列）：只需要边界 mask（N 非整数倍时）
    FA1 的做法是全程带 mask，白白多做了一倍的元素级比较。
    """
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo = start_m * BLOCK_M
        hi = (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    else:
        lo, hi = 0, N_CTX

    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))

    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # ---- S = Q K^T ----
        k = tl.load(K_block_ptr)  # (HEAD_DIM, BLOCK_N)，越界补 0
        qk = tl.dot(q, k)  # (BLOCK_M, BLOCK_N) fp32

        if STAGE == 2:
            # 对角块：j <= i 可见；j > i 置 -inf。越界列必然 > offs_m（因为
            # offs_m < N_CTX），会被同一个比较顺手 mask 掉，无需额外判断。
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0.0, _NEG)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk = qk - m_ij[:, None]
        else:
            qk = qk * qk_scale
            if not EVEN_N:
                # 尾部块：把 >= N_CTX 的列压成 -inf，否则补的 0 会当成"有效且
                # 相似度为 0"的 key，悄悄污染 softmax 分母。
                cols_ok = (start_n + offs_n)[None, :] < N_CTX
                qk = tl.where(cols_ok, qk, _NEG)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk = qk - m_ij[:, None]

        # ---- online softmax ----
        p = tl.math.exp2(qk)
        alpha = tl.math.exp2(m_i - m_ij)  # 旧累加器的重缩放因子
        l_ij = tl.sum(p, 1)
        acc = acc * alpha[:, None]

        # ---- O += P V ----
        v = tl.load(V_block_ptr)  # (BLOCK_N, HEAD_DIM)，越界补 0
        acc = tl.dot(p.to(v.dtype), v, acc)

        l_i = l_i * alpha + l_ij
        m_i = m_ij
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    return acc, l_i, m_i


@triton.jit
def _flash_attn_fwd(
    Q,
    K,
    V,
    sm_scale,
    Out,
    L,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,
    Z,
    H,
    N_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    EVEN_N: tl.constexpr,  # N_CTX 是否整除 BLOCK_N，由宿侧算好传入
):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    # (B,H,N,D) contiguous 时 z*stride_z + h*stride_h 可合并为 off_hz*stride_h
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh

    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(HEAD_DIM, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_vn, stride_vk),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    qk_scale = sm_scale * LOG2E
    q = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")

    if IS_CAUSAL:
        acc, l_i, m_i = _attn_fwd_inner(
            acc, l_i, m_i, q, K_block_ptr, V_block_ptr, start_m, qk_scale,
            BLOCK_M, HEAD_DIM, BLOCK_N, 1, offs_m, offs_n, N_CTX, EVEN_N,
        )
        acc, l_i, m_i = _attn_fwd_inner(
            acc, l_i, m_i, q, K_block_ptr, V_block_ptr, start_m, qk_scale,
            BLOCK_M, HEAD_DIM, BLOCK_N, 2, offs_m, offs_n, N_CTX, EVEN_N,
        )
    else:
        acc, l_i, m_i = _attn_fwd_inner(
            acc, l_i, m_i, q, K_block_ptr, V_block_ptr, start_m, qk_scale,
            BLOCK_M, HEAD_DIM, BLOCK_N, 3, offs_m, offs_n, N_CTX, EVEN_N,
        )

    # 最后统一除一次 l_i，而不是每步都除（少 BLOCK_N 次除法）
    acc = acc / l_i[:, None]
    # logsumexp（base-2 域），供后续融合或调试使用
    l = m_i + tl.math.log2(l_i)
    tl.store(L + off_hz * N_CTX + offs_m, l, mask=offs_m < N_CTX)
    tl.store(O_block_ptr, acc.to(Out.dtype.element_ty), boundary_check=(0,))


# sm_89 (Ada) 上的经验配置。
#
# 两处真实约束，不是拍脑袋：
#  1) shared memory。每 SM 只有 ~99 KB（101376 B）可用，K/V 的 tile 要乘
#     num_stages 份（软件流水），所以 BLOCK_N 一大就 OutOfResources。
#     D=256 时 BLOCK_N=64 + stages=3 实测需要 212992 B，直接超限。
#  2) 寄存器。BLOCK_M 大 → 累加器 acc (BLOCK_M × HEAD_DIM, fp32) 吃掉寄存器，
#     occupancy 掉下来反而更慢；这是 BLOCK_M 没有取更大的原因。
_CONFIGS = {
    16: {"BLOCK_M": 128, "BLOCK_N": 64, "num_stages": 3},
    32: {"BLOCK_M": 128, "BLOCK_N": 64, "num_stages": 3},
    64: {"BLOCK_M": 128, "BLOCK_N": 64, "num_stages": 3},
    128: {"BLOCK_M": 128, "BLOCK_N": 64, "num_stages": 3},
    # D=256（Gemma-2 这一类）必须缩 tile，否则 shared memory 超限
    256: {"BLOCK_M": 64, "BLOCK_N": 32, "num_stages": 2},
}


def flash_attn_forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = True):
    """q,k,v: (B,H,N,D) contiguous，dtype fp16/bf16/fp32。返回 (out, lse)。

    lse 形状 (B,H,N)，base-2 域（= log2(sum(exp2(qk*scale)))）。
    """
    assert q.is_cuda, "需要 CUDA"
    assert q.shape == k.shape == v.shape, f"形状必须一致: {q.shape} {k.shape} {v.shape}"
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    B, H, N, HEAD_DIM = q.shape
    if HEAD_DIM not in _CONFIGS:
        raise NotImplementedError(
            f"head_dim={HEAD_DIM} 未调优；在 _CONFIGS 里加一项即可（BLOCK_N <= HEAD_DIM）"
        )

    cfg = _CONFIGS[HEAD_DIM]
    out = torch.empty_like(q)
    lse = torch.empty((B, H, N), device=q.device, dtype=torch.float32)
    grid = (triton.cdiv(N, cfg["BLOCK_M"]), B * H, 1)
    _flash_attn_fwd[grid](
        q, k, v, 1.0 / math.sqrt(HEAD_DIM), out, lse,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        B, H, N,
        HEAD_DIM=HEAD_DIM,
        BLOCK_M=cfg["BLOCK_M"],
        BLOCK_N=cfg["BLOCK_N"],
        IS_CAUSAL=causal,
        EVEN_N=(N % cfg["BLOCK_N"] == 0),
        num_stages=cfg["num_stages"],
    )
    return out, lse


# ---------------------------------------------------------------- 参考实现


def naive_attention(q, k, v, causal: bool = True):
    """最朴素实现：真的把 N×N 的 S 物化出来。

    它不是"错误实现"，是"基准实现"——数值最可信，用来当 oracle 验证 Triton 版；
    同时也是 benchmark 里 O(N²) 显存那条曲线（N=8192 时它会直接 OOM，这正是
    FlashAttention 存在的理由）。
    """
    d = q.shape[-1]
    scale = 1.0 / math.sqrt(d)
    s = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    if causal:
        n = s.shape[-1]
        mask = torch.triu(torch.ones(n, n, device=s.device, dtype=torch.bool), diagonal=1)
        s = s.masked_fill(mask, float("-inf"))
    p = torch.softmax(s, dim=-1)
    return torch.matmul(p, v.float()).to(q.dtype)


def sdpa_attention(q, k, v, causal: bool = True):
    """Torch 自带 fused attention（派发到 flash / mem-efficient / math 后端）。

    这是"工业基线"：自己写的 kernel 连它都打不过，就不能拿"比朴素实现快"当成果。
    """
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal)