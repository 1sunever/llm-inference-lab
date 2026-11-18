"""投机解码的验证器：拒绝采样 + 残差修正。

参考：
  [1] Leviathan, Kalman, Matias. "Fast Inference from Transformers via
      Speculative Decoding." ICML 2023. arXiv:2211.17192
  [2] Chen, Borgeaud, Irving, Lespiau, Sifre, Jumper. "Accelerating Large
      Language Model Decoding with Speculative Sampling." arXiv:2302.01318

为什么这个文件是"最该写对的地方"
--------------------------------------------------------------------------------
投机解码的卖点是"白拿加速"，但它成立的唯一前提是：

    **用草稿 q 起草 + 用目标 p 验证，最终吐出的 token 的分布与
      直接对 p 采样完全一致（逐 token 精确相等，不是近似）。**

如果这一点不成立，那么投机解码就变成了"用质量换速度"的量化/蒸馏手段，
模型的行为会被悄悄改变（长尾 token 被系统性压低、重复度上升、pass@k 掉点），
这类回归在离线评测里极难发现，所以生产系统（vLLM / TensorRT-LLM / SGLang）
只接受分布无损的版本。下面把证明写在代码旁边，方便对照实验 A 一起看。

单步（一个位置）的证明
--------------------------------------------------------------------------------
记号：草稿分布 q，目标分布 p，草稿 token x ~ q。验证规则：
  1) 以概率 min(1, p(x)/q(x)) 接受 x；
  2) 拒绝时，从"修正分布" r(y) = norm(max(0, p(y) - q(y))) 采样一个 token 顶上。

那么输出 Y 的分布是：
  P(Y = y) = q(y) * min(1, p(y)/q(y))                    # 接受分支
           + P(reject) * r(y)                             # 拒绝分支

先算拒绝概率。注意 min(q, p) + max(0, q - p) = q，对 y 求和：
  P(accept) = Σ_y q(y) min(1, p(y)/q(y)) = Σ_y min(q(y), p(y))
            = 1 - Σ_y max(0, q(y) - p(y))
所以
  P(reject) = Σ_y max(0, q(y) - p(y))                     # (A)

修正分布的分母是 Σ_z max(0, p(z) - q(z))。而
  Σ_z (p(z) - q(z)) = 0  =>  Σ_z max(0, p(z) - q(z)) = Σ_z max(0, q(z) - p(z))   # (B)
也就是说式 (A) 的分子/分母之和恰好相等（都等于全变差距离 TV(p, q)），于是
  P(reject) * r(y) = Σ_max(q-p) * max(0, p(y)-q(y)) / Σ_max(p-q) = max(0, p(y) - q(y))。

代回：
  P(Y = y) = min(p(y), q(y)) + max(0, p(y) - q(y)) = p(y).  ∎

每一步都精确等于 p，而自回归解码的下一步只依赖已生成的 token，
所以按归纳法整条序列的分布也精确等于"直接对目标采样"。这就是它可以
安全地替换掉现有采样器的唯一理由。

推论（也是实验 A 的判据）：单步接受率的最大期望是
  E[accept] = Σ_y min(p(y), q(y)) = 1 - TV(p, q)，
`expected_acceptance_rate` 实现了它；当 q == p 时接受率 100%（退化但正确）。

实现注意
--------------------------------------------------------------------------------
* 全部用 float32 计算：fp16 下 `p - q` 的相减在 p≈q 时会被舍入误差吃光，
  残差分布会算错（尤其在 q 和 p 几乎一致的后期，恰好是最需要它正确的区间）。
* q(y) == 0 而 p(y) > 0 时比值是 inf，正确语义是"必然接受"。
* 数值上 max(0, p - q) 可能全为 0（p 与 q 完全相等时的舍入产物）；
  此时退回目标分布，不改变正确性（该情形下 P(reject) 本身也是 0）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Union

import torch

__all__ = [
    "VerifyResult",
    "speculative_verify",
    "speculative_verify_batch",
    "naive_accept_draft",
    "naive_verify_no_correction_batch",
    "expected_accepted_tokens",
    "expected_speedup",
    "optimal_k",
    "expected_acceptance_rate",
    "distribution_distance",
]

TensorLike = Union[torch.Tensor, Sequence, Sequence[Sequence]]


# --------------------------------------------------------------------------- #
# 单条序列的参考实现（可读性优先，实验 C / 单测用它）
# --------------------------------------------------------------------------- #


@dataclass
class VerifyResult:
    """一轮"起草 -> 验证"产出的东西。

    tokens          本轮真正写进输出序列的 token（已接受的草稿 + 1 个额外 token）
    num_accepted    其中有多少个来自草稿（不含 bonus / 修正 token）
    used_correction 是否发生了拒绝（True 时最后一个 token 来自修正分布）
    bonus_token     全部接受时从 target 额外采的 token（None 表示没有 bonus）
    """

    tokens: List[int] = field(default_factory=list)
    num_accepted: int = 0
    used_correction: bool = False
    bonus_token: Optional[int] = None

    @property
    def num_emitted(self) -> int:
        """本轮净产出的 token 数 = num_accepted + 1（修正 token 或 bonus）。"""
        return len(self.tokens)


def _gen_device(generator: Optional[torch.Generator], default: torch.device) -> torch.device:
    """随机数发生器所在的设备。

    torch 要求 `multinomial/rand` 的概率张量与 generator 在同一设备上，
    否则抛 "Expected a 'cuda' device type for generator"。真实模型那条路径上
    概率张量在 GPU 上，所以生成器也得是 CUDA 的（否则每步多一次 3.6MB 的
    H2D/D2H 拷贝，会把验证开销算到投机解码头上）。
    """
    if generator is None:
        return default
    dev = getattr(generator, "device", None)
    return default if dev is None else torch.device(dev)


def _as_probs2d(x: TensorLike, name: str) -> torch.Tensor:
    t = torch.as_tensor(x, dtype=torch.float32)
    if t.dim() != 2:
        raise ValueError(f"{name} 必须是二维 [k, V]，实际 shape={tuple(t.shape)}")
    if not torch.isfinite(t).all():
        raise ValueError(f"{name} 含 nan/inf")
    if (t < 0).any():
        raise ValueError(f"{name} 含负概率")
    row_sum = t.sum(dim=-1)
    if not torch.allclose(row_sum, torch.ones_like(row_sum), atol=1e-3):
        raise ValueError(f"{name} 的行没有归一化，行和={row_sum.tolist()}")
    return t


def speculative_verify(
    draft_probs: TensorLike,
    target_probs: TensorLike,
    draft_tokens: Sequence[int],
    generator: Optional[torch.Generator] = None,
) -> VerifyResult:
    """对一条草稿序列做拒绝采样验证（参考实现，逐位置串行）。

    参数
    ----
    draft_probs   [k, V] 草稿模型在 k 个位置上的分布 q_i
    target_probs  [k, V] 或 [k+1, V]  目标模型分布 p_i。
                  多出来的第 k 行是"全部接受后额外采 bonus token"用的分布，
                  它应当来自目标模型对第 k 个草稿位置的下一步预测。
                  prompt-lookup 这类确定性草稿器可以只给 [k, V]（不要 bonus），
                  也可以按"草稿是 one-hot"构造出第 k 行来拿 bonus。
    draft_tokens  [k] 草稿 token id
    generator     torch.Generator，用于可复现

    返回
    ----
    VerifyResult

    语义：位置 i 的接受概率是 min(1, p_i(x_i) / q_i(x_i))；第一个被拒的位置
    用 norm(max(0, p_i - q_i)) 采样一个 token 顶替并结束本轮；全部接受则从
    target_probs[k]（若提供）再采一个 bonus token。见文件头部的分布等价证明。
    """
    q = _as_probs2d(draft_probs, "draft_probs")
    p = _as_probs2d(target_probs, "target_probs")
    dev = _gen_device(generator, q.device)
    q, p = q.to(dev), p.to(dev)
    k = q.shape[0]
    if p.shape[1] != q.shape[1]:
        raise ValueError(f"词表大小不一致: q={q.shape[1]} p={p.shape[1]}")
    if p.shape[0] not in (k, k + 1):
        raise ValueError(f"target_probs 行数应为 k={k} 或 k+1={k + 1}，实际 {p.shape[0]}")
    toks = [int(t) for t in draft_tokens]
    if len(toks) != k:
        raise ValueError(f"draft_tokens 长度 {len(toks)} != k={k}")

    result = VerifyResult()
    for i in range(k):
        x = toks[i]
        if not 0 <= x < q.shape[1]:
            raise ValueError(f"draft token {x} 越界")
        p_x = float(p[i, x])
        q_x = float(q[i, x])
        # q_x == 0 -> 草稿不可能生成这个 token，接受概率按 1 处理（inf 的极限）。
        ratio = 1.0 if q_x <= 0.0 else min(1.0, p_x / q_x)
        u = float(torch.rand((), generator=generator, device=dev))
        if u < ratio:
            result.tokens.append(x)
            result.num_accepted += 1
            continue

        # ---- 第一个被拒的位置：用修正分布 norm(max(0, p - q)) 重新采一个 ----
        # 注意这里必须减掉 q：如果直接对 p 采样（少了这一步），输出分布会变成
        # p + (接受质量) 的混合，系统性高估 p 高的位置、低估 p 低的位置。
        residual = torch.clamp(p[i] - q[i], min=0.0)
        total = float(residual.sum())
        if total <= 0.0:
            # 数值退化（p 与 q 在 float32 下完全相等）：理论上 P(reject)=0 不会走到这，
            # 保险起见退回目标分布本身，不引入偏差。
            residual = p[i]
            total = float(residual.sum())
        correction = int(torch.multinomial(residual / total, 1, generator=generator))
        result.tokens.append(correction)
        result.used_correction = True
        return result

    # ---- 全部接受：从目标分布再采一个 bonus token ----
    if p.shape[0] == k + 1:
        bonus = int(torch.multinomial(p[k], 1, generator=generator))
        result.tokens.append(bonus)
        result.bonus_token = bonus
    return result


# --------------------------------------------------------------------------- #
# 批量实现：同一条算法，但把"试验"放到第 0 维用张量并行
# （实验 A 要 20 万次采样，串行版本在 Python 里太慢；两者的分布必须一致，
#   实验 A 里用同一个检验同时验证了这一点）
# --------------------------------------------------------------------------- #


def _batch_verify_core(
    draft_probs: torch.Tensor,
    target_probs: torch.Tensor,
    draft_tokens: torch.Tensor,
    generator: Optional[torch.Generator],
    use_residual: bool,
) -> dict:
    """批量验证的核心循环。use_residual=False 时是"错误版本"，仅供对照实验。"""
    q = torch.as_tensor(draft_probs, dtype=torch.float32)
    p = torch.as_tensor(target_probs, dtype=torch.float32)
    x = torch.as_tensor(draft_tokens, dtype=torch.long)
    dev = _gen_device(generator, q.device)
    q, p, x = q.to(dev), p.to(dev), x.to(dev)
    if q.dim() != 3:
        raise ValueError(f"draft_probs 需要 [B, k, V]，实际 {tuple(q.shape)}")
    B, k, V = q.shape
    if tuple(x.shape) != (B, k):
        raise ValueError(f"draft_tokens 需要 [B={B}, k={k}]，实际 {tuple(x.shape)}")
    if p.dim() != 3 or p.shape[0] != B or p.shape[2] != V or p.shape[1] not in (k, k + 1):
        raise ValueError(f"target_probs 需要 [B, k 或 k+1, V]，实际 {tuple(p.shape)}")

    emitted = torch.full((B, k + 1), -1, dtype=torch.long, device=dev)
    num_accepted = torch.zeros(B, dtype=torch.long, device=dev)
    used_correction = torch.zeros(B, dtype=torch.bool, device=dev)
    alive = torch.ones(B, dtype=torch.bool, device=dev)  # 还没被拒的序列

    for i in range(k):
        xi = x[:, i]
        p_i = p[:, i].gather(1, xi[:, None]).squeeze(1)
        q_i = q[:, i].gather(1, xi[:, None]).squeeze(1)
        # q_i == 0 -> 必然接受（p/q 的极限）
        ratio = torch.where(
            q_i > 0, torch.clamp(p_i / q_i.clamp_min(1e-30), max=1.0), torch.ones_like(p_i)
        )
        u = torch.rand(B, generator=generator, device=dev)
        accept = alive & (u < ratio)

        if bool(accept.any()):
            idx = accept.nonzero(as_tuple=True)[0]
            emitted[idx, num_accepted[idx]] = xi[idx]
            num_accepted[idx] += 1

        reject = alive & ~accept
        if bool(reject.any()):
            idx = reject.nonzero(as_tuple=True)[0]
            if use_residual:
                cand = torch.clamp(p[idx, i] - q[idx, i], min=0.0)
            else:
                # 反例：忘了减 q，直接对 p 采样
                cand = p[idx, i]
            total = cand.sum(dim=-1, keepdim=True)
            dead = total.squeeze(-1) <= 0
            if bool(dead.any()):
                cand = torch.where(dead[:, None], p[idx, i], cand)
                total = cand.sum(dim=-1, keepdim=True)
            sampled = torch.multinomial(cand / total, 1, generator=generator).squeeze(1)
            emitted[idx, num_accepted[idx]] = sampled
            used_correction[idx] = True
            alive[idx] = False

    lengths = num_accepted.clone()
    lengths[used_correction] += 1  # 修正 token 也占一个输出位置
    if p.shape[1] == k + 1 and bool(alive.any()):
        idx = alive.nonzero(as_tuple=True)[0]
        bonus = torch.multinomial(p[idx, k], 1, generator=generator).squeeze(1)
        emitted[idx, num_accepted[idx]] = bonus
        lengths[idx] += 1
    return {
        "tokens": emitted,          # [B, k+1]，-1 是没用到的槽位
        "lengths": lengths,         # [B] 本轮实际输出多少 token
        "num_accepted": num_accepted,   # [B] 接受了几个草稿 token
        "used_correction": used_correction,
    }


def speculative_verify_batch(
    draft_probs: torch.Tensor,
    target_probs: torch.Tensor,
    draft_tokens: torch.Tensor,
    generator: Optional[torch.Generator] = None,
) -> dict:
    """批量版的拒绝采样验证（与 `speculative_verify` 算法逐字相同）。

    参数都是带 batch 维的张量：draft_probs [B,k,V]，target_probs [B,k(+1),V]，
    draft_tokens [B,k]。返回 dict：tokens [B,k+1]（-1 填充）、lengths [B]、
    num_accepted [B]、used_correction [B]。

    实现上和串行版的唯一区别是"哪一维在循环"：这里循环 k 个位置，
    所有试验在第 0 维上一起做。分布性质完全一样（同一个 min(1,p/q) 接受规则、
    同一个残差修正），实验 A 用 20 万条样本同时验证了这两件事。
    """
    return _batch_verify_core(draft_probs, target_probs, draft_tokens, generator, use_residual=True)


def naive_accept_draft(draft_tokens: torch.Tensor) -> torch.Tensor:
    """反例 1：不做任何验证，直接接受草稿 token。

    输出分布 = q，而不是 p。偏差大小 = TV(p, q)。这是"草稿模型差一点也没关系"
    这种直觉的来源，也正是投机解码必须拒绝的做法。
    """
    return torch.as_tensor(draft_tokens, dtype=torch.long)


def naive_verify_no_correction_batch(
    draft_probs: torch.Tensor,
    target_probs: torch.Tensor,
    draft_tokens: torch.Tensor,
    generator: Optional[torch.Generator] = None,
) -> dict:
    """反例 2（更隐蔽）：接受规则对，但拒绝时忘了减 q。

    拒绝分支本应采 norm(max(0, p - q))，这里直接采 p。看起来"被拒了就重新
    按目标分布采"很合理，实际效果是接受分支的 min(p,q) 质量和拒绝分支的 p
    质量叠加，输出分布 = min(p,q) + P(reject)·p ≠ p：
      * p 高 q 也高的位置被高估（接受的那份 + 拒绝重采的那份）；
      * p 低的位置被低估。
    实验 A 会给出它的 L1 距离作为量化证据。
    """
    return _batch_verify_core(draft_probs, target_probs, draft_tokens, generator, use_residual=False)


# --------------------------------------------------------------------------- #
# 解析公式：期望 token 数 / 加速比 / 最优 k
# --------------------------------------------------------------------------- #


def expected_accepted_tokens(alpha: float, k: int) -> float:
    """一轮里期望产出多少个 token：1 + α + α² + ... + α^k。

    推导（[1] 的 §3.1，几何级数）：
      设每个草稿 token 独立地以概率 α 被接受，A = 被接受的个数。
      对非负整数 A 有 E[A] = Σ_{j≥1} P(A ≥ j)，而 "至少接受 j 个" 等价于
      前 j 个都接受，概率 α^j。于是
          E[A] = α + α² + ... + α^k。
      一轮无论如何还会多产出 1 个 token（全部接受时是 target 的 bonus token，
      有拒绝时是修正 token），所以
          E[tokens/轮] = 1 + α + α² + ... + α^k = (1 - α^{k+1}) / (1 - α)   (α < 1)

    退化情形 α == 1：k 个草稿全接受 + 1 个 bonus = k + 1，公式的极限一致
    （分子分母同时为 0，取极限得 k+1；这里显式返回，避免 0/0）。
    α == 0：只有 1 个 token（草稿全被拒），公式同样给出 1。
    """
    alpha = float(alpha)
    k = int(k)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha 必须在 [0,1]，实际 {alpha}")
    if k < 0:
        raise ValueError(f"k 必须 >= 0，实际 {k}")
    if alpha == 1.0:
        return float(k + 1)
    return (1.0 - alpha ** (k + 1)) / (1.0 - alpha)


def expected_speedup(alpha: float, k: int, cost_ratio: float) -> float:
    """期望加速比 = 每轮产出的 token 数 / 每轮的墙钟成本。

    把"目标模型一次前向（验证 k 个草稿 token）"的成本记为 1，草稿模型单步成本记为
    cost_ratio（= 草稿单步 / 目标单步）。一轮的成本是 1 + k * cost_ratio，于是
        speedup = (1 + α + ... + α^k) / (1 + k·cost_ratio)

    两个极限值得记住：
      * k=0：退化成普通自回归，speedup = 1/(1+0) = 1；
      * k→∞：(1/(1-α)) / (1 + k·c) → 0。分母线性增长、分子饱和，
        所以 **k 太大一定变慢**——多出来的草稿全是白算（见 optimal_k 与实验 B 的拐点）。
    注意这里忽略了验证 k 个 token 比验证 1 个 token 略贵（注意力更长、KV 写入更多），
    真实系统的 cost_ratio 要按实测填，实验 C 就是这么做的。
    """
    cost_ratio = float(cost_ratio)
    if cost_ratio < 0:
        raise ValueError(f"cost_ratio 必须 >= 0，实际 {cost_ratio}")
    return expected_accepted_tokens(alpha, k) / (1.0 + k * cost_ratio)


def optimal_k(alpha: float, cost_ratio: float, k_max: int = 64) -> dict:
    """使 `expected_speedup` 最大的 k（数值扫描；对 k 求导含 α^k 项，无简洁闭式解）。

    返回 {"k": 最优 k, "speedup": 峰值加速比, "k_max": 扫描上界}。
    典型形状：k 从 1 开始涨、到某个点后掉头向下——那个点就是拐点。
    """
    if k_max < 1:
        raise ValueError("k_max 必须 >= 1")
    best_k, best_v = 0, expected_speedup(alpha, 0, cost_ratio)
    for k in range(1, int(k_max) + 1):
        v = expected_speedup(alpha, k, cost_ratio)
        if v > best_v:
            best_k, best_v = k, v
    return {"k": best_k, "speedup": best_v, "k_max": int(k_max)}


def expected_acceptance_rate(p: TensorLike, q: TensorLike) -> float:
    """q 起草、p 验证时的期望接受率 = Σ_y min(p(y), q(y)) = 1 - TV(p, q)。

    这是分布等价性的副产品：q 越接近 p，接受率越接近 1。实验 A 会拿它跟
    20 万次采样的经验接受率对比（两者必须一致）。
    """
    pt = torch.as_tensor(p, dtype=torch.float32).flatten()
    qt = torch.as_tensor(q, dtype=torch.float32).flatten()
    if pt.shape != qt.shape:
        raise ValueError("p 与 q 形状必须一致")
    return float(torch.minimum(pt, qt).sum())


# --------------------------------------------------------------------------- #
# 实验 A 用的度量
# --------------------------------------------------------------------------- #


def distribution_distance(
    empirical: TensorLike, target: TensorLike, counts: Optional[TensorLike] = None
) -> dict:
    """经验分布 vs 目标分布：L1 / 最大绝对误差 / 卡方统计量。

    empirical 给频率（已归一化）或计数；给计数时必须同时给 n（用 counts 参数传样本数）
    才能算卡方。这里统一要求传"频率"，卡方用 n_eff * freq 还原观测计数。

    L1 距离（全变差的两倍）对采样噪声最敏感也最好解释：
    正确实现的 L1 只受 ~sqrt(p(1-p)/n) 的噪声限制；错误实现会偏离到 TV 量级。
    """
    e = torch.as_tensor(empirical, dtype=torch.float64).flatten()
    t = torch.as_tensor(target, dtype=torch.float64).flatten()
    if e.shape != t.shape:
        raise ValueError("empirical 与 target 形状必须一致")
    if e.sum() > 0:
        e = e / e.sum()
    n = float(counts) if counts is not None else None
    out = {
        "l1": float((e - t).abs().sum()),
        "max_abs_err": float((e - t).abs().max()),
    }
    if n is not None and n > 0:
        obs = e * n
        exp = t * n
        out["chi2"] = float((((obs - exp) ** 2) / exp.clamp_min(1e-12)).sum())
        out["dof"] = int(t.numel() - 1)
        # 自由度 7、显著性 0.05 的卡方临界值（scipy 不在依赖里，查表即可）
        critical = {1: 3.841, 2: 5.991, 3: 7.815, 4: 9.488, 5: 11.070, 6: 12.592, 7: 14.067}
        out["chi2_critical_0.05"] = critical.get(out["dof"])
    return out