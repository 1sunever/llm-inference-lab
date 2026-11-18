"""MoE 层：一个 SwiGLU expert + 一个 dense 对照 + 用 scatter 拼起来的 MoE 层。

参数/算力的账（dim=D, expert 隐层=H, E 个 expert, 每 token 取 k 个）：

    SwiGLU 一个 expert  = w1[D,H] + w3[D,H] + w2[H,D] = 3DH 个参数
    MoELayer 总参数     = E * 3DH (+ 路由的 D*E，占比 H 量级的 1/(3H) 的零头)
    同参数量 DenseFFN   = 隐层取 E*H，即 3D(EH)，每 token 全部激活
    MoE 每 token 算力   ∝ k * 3DH

  → 同样显存（约 E 倍参数）下，MoE 每 token 只花 k/E 的算力。
    这就是 MoE 的全部卖点：**显存/参数按 E 买，算力按 k 付**。
  → 但反过来看，MoE 并不省"每 token 要读的权重"这件事的带宽下限：
    batch=1 时它读 k 个 expert 的权重（k*3DH*4 字节），同参数量 dense 读 E 份，
    所以 MoE 在 decode 时理论上该更快；可实测常常相反，原因是
    (a) 路由本身要算 gate + topk + 排序 + 容量掩码（好几个小 kernel），
    (b) gather/scatter 是随机的全局访存，命中率极差，
    (c) E 份小 GEMM 的 launch 开销 + 每份只有 1 个 token，SM 基本空转。
    谁赢就是 bench.py A 组要测的东西，答案不该由我先写。

scatter 的实现约束（题目要求 + 性能要求）：
  不写 `for e in range(num_experts)` 逐 expert 取子集算。E=32 时那是 96 次 kernel
  launch，在 batch=1 上比算术还贵。这里走"pad 成 [E, cap, D] 的槽位张量 + 一次 bmm"：
    1. index_add_ 把 token 散射进各自 expert 的槽位（dropless 的 scatter）；
    2. 三个 bmm 对 E 个 expert 一次性做 SwiGLU（等价于 grouped GEMM，只是用 pad 换规整）；
    3. index_select + 加权 + index_add_ 收回来。
  代价是 padding：cap 必须取所有 expert 里的最大负载（dropless）或声明的 capacity
  （有容量限制时）。这份 padding 是**真的会被算的**，所以实验 B 里 capacity_factor
  变大时 moe_ms 会真的变长——那不是模拟出来的浪费，是 bmm 真算在了零行上。
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .router import TopKRouter, _position_in_expert


class SwiGLUExpert(nn.Module):
    """一个 expert = 一个 SwiGLU FFN： w2( silu(w1 x) * (w3 x) )。

    SwiGLU（Shazeer 2020）相对 ReLU-FFN 的两个 gate：
      w1 是"值"通路，w3 是"门"通路，silu 作用在门上是平滑的乘性门控，
      同参数量下通常比 GELU-FFN 好，代价是 3 个矩阵而不是 2 个。
    参数按 [in, out] 存（和 PyTorch Linear 一致），这样 x @ w 不需要转置。
    """

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.dim, self.hidden = int(dim), int(hidden)
        self.w1 = nn.Parameter(torch.empty(self.dim, self.hidden))
        self.w3 = nn.Parameter(torch.empty(self.dim, self.hidden))
        self.w2 = nn.Parameter(torch.empty(self.hidden, self.dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # 与 nn.Linear 默认一致的截断正态，std=1/sqrt(fan_in)
        for w, fan_in in ((self.w1, self.dim), (self.w3, self.dim), (self.w2, self.hidden)):
            nn.init.normal_(w, std=fan_in ** -0.5)

    @property
    def num_params(self) -> int:
        return self.w1.numel() + self.w3.numel() + self.w2.numel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., dim] -> [..., dim]。"""
        return (F.silu(x @ self.w1) * (x @ self.w3)) @ self.w2


class DenseFFN(nn.Module):
    """等参数量的 dense 对照：结构上就是 SwiGLUExpert，换一个隐层宽度。

    做"等参数量"对照时传 hidden = num_experts * moe_hidden（见 bench.py），
    这时它和 MoELayer 吃一样多的显存，但每 token 全部激活。
    """

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.dim, self.hidden = int(dim), int(hidden)
        self.w1 = nn.Parameter(torch.empty(self.dim, self.hidden))
        self.w3 = nn.Parameter(torch.empty(self.dim, self.hidden))
        self.w2 = nn.Parameter(torch.empty(self.hidden, self.dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for w, fan_in in ((self.w1, self.dim), (self.w3, self.dim), (self.w2, self.hidden)):
            nn.init.normal_(w, std=fan_in ** -0.5)

    @property
    def num_params(self) -> int:
        return self.w1.numel() + self.w3.numel() + self.w2.numel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (F.silu(x @ self.w1) * (x @ self.w3)) @ self.w2


def dense_hidden_for_params(dim: int, target_params: int) -> int:
    """反解"要多少隐层宽度的 DenseFFN 才能和 MoELayer 参数量对齐"。

    3*D*H_d = target  →  H_d = target / (3D)。
    向上取整会让 dense 略多几百个参数，bench 里会把两边的精确参数量都记进 JSON，
    不允许"约等于"含糊过去。
    """
    return max(1, int(round(target_params / (3.0 * dim))))


class MoELayer(nn.Module):
    """路由 → 分桶 scatter → 一次 bmm 算 E 个 expert → 加权收回。

    不内建 residual：返回的是"本层的增量"，调用方写 x = x + moe(x)。
    被容量丢光 slot 的 token 拿到精确的 0 增量 = 残差直通，这正是 GShard 的语义。
    """

    def __init__(
        self,
        dim: int,
        hidden: int,
        num_experts: int,
        top_k: int = 2,
        capacity_factor: float = 1.25,
        dropless: bool = False,
        jitter_noise: float = 0.0,
    ):
        super().__init__()
        self.dim = int(dim)
        self.hidden = int(hidden)
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.dropless = bool(dropless)

        self.router = TopKRouter(
            hidden_dim=self.dim,
            num_experts=self.num_experts,
            top_k=self.top_k,
            capacity_factor=capacity_factor,
            dropless=dropless,
            jitter_noise=jitter_noise,
        )

        # expert 权重按 [E, in, out] 堆叠：这样 E 个 expert 的 GEMM 是一次 bmm，
        # 而不是 E 次 addmm。等价于 grouped GEMM 的"权重连续"前提。
        self.w1 = nn.Parameter(torch.empty(self.num_experts, self.dim, self.hidden))
        self.w3 = nn.Parameter(torch.empty(self.num_experts, self.dim, self.hidden))
        self.w2 = nn.Parameter(torch.empty(self.num_experts, self.hidden, self.dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for w, fan_in in ((self.w1, self.dim), (self.w3, self.dim), (self.w2, self.hidden)):
            nn.init.normal_(w, std=fan_in ** -0.5)

    # ------------------------------------------------------------ 记账

    @property
    def num_params(self) -> int:
        """含路由。"""
        return (
            self.w1.numel() + self.w3.numel() + self.w2.numel() + self.router.weight.numel()
        )

    @property
    def num_params_experts(self) -> int:
        return self.w1.numel() + self.w3.numel() + self.w2.numel()

    @property
    def active_params_per_token(self) -> int:
        """每 token 真正被读到的参数量 ≈ k 个 expert + 一遍 gate。

        与 num_params 的比值就是"稀疏度"；这个数和 FLOPs 成正比，
        面试里问"MoE 省了什么"应该直接报这两个数。
        """
        return self.top_k * (3 * self.dim * self.hidden) + self.router.weight.numel()

    # ------------------------------------------------------------ 前向

    def forward(
        self,
        x: torch.Tensor,
        collect_metrics: bool = True,
        compute_aux: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """x: [num_tokens, dim]（3D 输入请调用方 flatten，见 bench）。

        collect_metrics / compute_aux 的语义见 TopKRouter.forward：
        纯推理计时关掉它们，少一次 D2H 同步和 6 个 aux kernel。

        Returns: (out [N, dim], aux_loss 标量 tensor, metrics dict)
        """
        if x.dim() != 2:
            raise ValueError(f"MoELayer 只吃 [N, dim]，收到 {tuple(x.shape)}")
        n_tok, D = x.shape
        E, k, H = self.num_experts, self.top_k, self.hidden

        dispatch, top_i, aux_loss, metrics = self.router(
            x, collect_metrics=collect_metrics, compute_aux=compute_aux
        )

        # 路由器把被丢弃的 slot 权重写成精确的 0.0，所以 keep 掩码可以直接从权重还原，
        # 不需要 router 多返回一个张量（也就不用为该掩码设计第 5 个返回值）。
        # 只用一次 nonzero（布尔索引一次 = 3 个 kernel：cumsum/select/write_indices），
        # 然后全部走 flat 的 index_select，避免对 (N,k) 做三次布尔掩码。
        sel = torch.nonzero((dispatch > 0).reshape(-1)).squeeze(1)     # 保留的分配，flat 下标
        n_kept = sel.numel()          # 注意：nonzero 本身就要一次 D2H（eager 无法避免）

        if n_kept == 0:
            # 极端情况（capacity_factor 极小或全 token 挤在一个 expert）：
            # 什么都不算，直接返回零增量 = 全量残差直通。
            out = torch.zeros_like(x)
            metrics["kept_assignments"] = 0
            return out, aux_loss, metrics

        exp_idx = top_i.reshape(-1).index_select(0, sel)               # [n_kept] 属于哪个 expert
        w_kept = dispatch.reshape(-1).index_select(0, sel)             # [n_kept] fp32 gate 权重
        tok_idx = torch.div(sel, k, rounding_mode="floor")             # [n_kept] 回推 token 下标

        # 队内序号：在保留的分配上重算，得到每个 expert 从 0 开始连续的槽位号。
        # （router 里的 rank 是在全量分配上算的，含被丢的；这里过滤后重算，
        #  顺便保证槽位号连续、不留洞，buffer 才能取 max(load)-1 那么小。）
        rank = _position_in_expert(exp_idx, E)

        cap = int(metrics["capacity_used"])         # dropless: 最大负载；否则: 声明的 capacity
        cap = max(cap, 1)
        n_slots = E * cap
        slot = exp_idx * cap + rank                 # 槽位线性下标，天然无重复

        # ---- 1) scatter：把 token 塞进 [E*cap, D] 的槽位缓冲 ----
        # index_add_ 而不是 index_copy_：语义上就是"多对一求和"的散射原语，
        # 这里 slot 唯一所以结果与赋值相同，但用它更贴近 dropless scatter 的定义。
        xin = x.new_zeros(n_slots, D).index_add_(0, slot, x.index_select(0, tok_idx))
        xin = xin.view(E, cap, D)

        # ---- 2) 一次 bmm 算完 E 个 expert ----
        # 注意：cap 里没被填满的槽位是零行，它们照样参与 GEMM —— 这就是容量因子
        # 带来的**真实** padding 浪费，实验 B 的 moe_ms 曲线会把它量出来。
        h = F.silu(torch.bmm(xin, self.w1)) * torch.bmm(xin, self.w3)   # [E, cap, H]
        y = torch.bmm(h, self.w2)                                       # [E, cap, D]

        # ---- 3) gather 回来，按 gate 权重加权求和 ----
        y = y.reshape(n_slots, D).index_select(0, slot)                 # [n_kept, D]
        y = y * w_kept.to(y.dtype).unsqueeze(-1)
        out = x.new_zeros(n_tok, D).index_add_(0, tok_idx, y)

        if collect_metrics:
            metrics["pad_slots"] = n_slots
            metrics["padding_rows"] = n_slots - n_kept
            # 真实算在零行上的比例：= 1 - 有效槽位利用率
            metrics["padding_compute_ratio"] = (n_slots - n_kept) / max(n_slots, 1)
        return out, aux_loss, metrics