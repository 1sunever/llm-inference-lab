"""Top-K 路由：MoE 里唯一"每个 token 都必须跑"的一层。

三笔账要先记住，后面 bench.py 的三个实验就是围着它们转的：

  1. gate 是个把 hidden 投到 num_experts 的小矩阵（dim × E），参数量占比极小，
     但它决定"哪几个 expert 的权重会被读进 SM"。路由错了 = 算力白花、负载歪掉。

  2. 归一化 top-k：把 k 个 prob 除以它们自己的和，使 dispatch 权重和为 1。
     这与原始 Switch Transformer 的 top-1 不同 ——
       - Switch top-1：dispatch 权重就是 gate 概率本身，和 < 1，
         等于用一个"置信度"整体缩放这一层的输出；
       - 归一化 top-k（Mixtral / GShard 后期做法，本文实现）：
         每 token 的输出幅度与 k 无关，训练更稳；
         代价是丢掉了"这个 token 有多确定"这一信号（反正 auxiliary loss 会另给）。
     本文件同时支持 k=1 与 k>1，注释里点明区别，方便对照论文读。

  3. aux loss 的梯度只走 P_i，不走 f_i：
       f_i = 被路由到 expert i 的 token 比例 —— 来自 argmax/topk，**不可导**；
       P_i = gate 概率在 token 维上的均值        —— 来自 softmax，**可导**。
     所以"负载均衡"这个约束不是通过改 argmax 实现的，而是通过把 softmax 的
     平均分布往均匀推来实现的：被冷落的 expert 的 P_i 变大，logit 慢慢被抬起来，
     下一轮才会有 token 真的路由过去。这是 aux loss 慢但有效的原因（见实验 C）。

容量（capacity）的语义，与 GShard/Switch 一致：
    capacity = ceil((num_tokens / num_experts) * capacity_factor)
  每个 expert 只吃前 capacity 个分给它的 token（按 token 在 batch 里的顺序排队），
  排到后面的**那个 slot** 被丢弃。注意 top_k>1 时丢弃是 per-slot 的：
  一个 token 的两个 slot 可能只丢一个，此时它的输出幅度会变小 —— 这里刻意
  **不**对剩余权重重新归一化，因为"被容量截断"在真实系统里就该表现为
  该 token 少算了一半，重新归一化会把这个信号藏起来。
  若一个 token 的 slot 全被丢光，它在本层的输出就是 0，等价于残差直通
  （MoELayer 不内建 residual，调用方写 x = x + moe(x)，0 就是直通）。
"""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _position_in_expert(expert_idx: torch.Tensor, num_experts: int) -> torch.Tensor:
    """每条 (token, slot) 分配在"它所属 expert 的队列"里排第几（0-based，按 token 顺序）。

    这是容量丢弃的核心：capacity 限制的是"每个 expert 前多少个 token 能进来"，
    所以必须知道每条分配的队内序号，且队内顺序 = token 在 batch 里的顺序
    （GShard 的 position-in-expert 语义）。

    实现刻意不写 `for e in range(num_experts)`——那会启动 E 次小 kernel，
    decode 场景里 launch 间隙比算术贵一个数量级。这里用 one-hot 的累加和：
      1. one-hot 出 [Nk, E]，沿 token 维做 cumsum：第 i 行第 e 列 = 到第 i 条分配为止
         有幾条分给了 e（token 顺序天然保住，因为 cumsum 的扫描顺序就是 token 顺序）；
      2. 在自己那一列上 gather 出来就是 1-based 队内序号，减 1 得 0-based。
    全程 3 个 kernel（one_hot / cumsum / gather），无排序、无 Python 循环。

    代价是 [Nk, E] 的中间张量：O(Nk·E) 内存换 O(Nk log Nk) 排序。
    E 是小量（本 repo 最大 32，实验里最大 Nk=8192 -> 1 MB），这笔换划算；
    如果是上亿 token 的预训练，会用 argsort(stable) + 分段扫描，省掉这个中间量。
    """
    flat = expert_idx.reshape(-1)
    oh = F.one_hot(flat, num_experts).to(torch.int32)          # [Nk, E]
    pos = torch.cumsum(oh, dim=0)                               # [Nk, E] token 顺序累加
    rank = pos.gather(1, flat.unsqueeze(1)).squeeze(1) - 1       # 1-based -> 0-based
    return rank.view_as(expert_idx)


class TopKRouter(nn.Module):
    """Top-k gate + 容量丢弃 + Switch 风格负载均衡 aux loss。

    Args:
        hidden_dim:      输入维度
        num_experts:     expert 数量
        top_k:           每个 token 送几个 expert
        capacity_factor: 容量因子，None/<=0 表示不设容量（等价 dropless）
        dropless:        True 时完全不丢弃（capacity 只作为参考值记录）
        jitter_noise:    训练时给 logits 乘上 U(1-eps, 1+eps) 的抖动（Switch 的技巧），
                         作用是让"同一个 token 每次前向路由结果略有不同"，
                         逼 expert 不能只记住固定 token 集。0 表示关闭。
    """

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        top_k: int = 2,
        capacity_factor: float = 1.25,
        dropless: bool = False,
        jitter_noise: float = 0.0,
    ):
        super().__init__()
        if not 1 <= top_k <= num_experts:
            raise ValueError(f"top_k={top_k} 必须落在 [1, num_experts={num_experts}]")

        self.hidden_dim = int(hidden_dim)
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.capacity_factor = float(capacity_factor)
        self.dropless = bool(dropless)
        self.jitter_noise = float(jitter_noise)

        # W_g: [hidden, num_experts]，按 "logits = x @ W_g" 布局直接存，
        # 而不是 nn.Linear 的 [E, hidden] 再转置 —— 省掉每步一次 .t()（它要 copy）。
        self.weight = nn.Parameter(torch.empty(self.hidden_dim, self.num_experts))
        nn.init.normal_(self.weight, std=self.hidden_dim ** -0.5)

        # 每个 expert 的 logit 偏置。默认全 0 不影响数学；
        # 两种真实用途：(a) 实验 B 用它注入热点，模拟 expert collapse；
        # (b) DeepSeek-V2 的 aux-loss-free 均衡就用这种可学习/可调 bias 替代 aux loss。
        self.register_buffer("expert_bias", torch.zeros(self.num_experts))

    # ------------------------------------------------------------ 前向

    def forward(
        self,
        x: torch.Tensor,
        collect_metrics: bool = True,
        compute_aux: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """x: [num_tokens, hidden_dim]（3D 由调用方 flatten）。

        Args:
            collect_metrics: False 时不算那些需要 .item() 的统计量。
                每个 .item() 都是一次 D2H 同步，会把 host 的延迟记到路由头上；
                bench A 组（纯推理计时）传 False，B/C 组要统计所以传 True。
            compute_aux: False 时跳过 aux loss 的那几个 kernel（profiler 实测 8 个）。
                纯推理没有 aux loss（它只为训练服务），bench A 组按推理场景关掉，
                同时另测一份 with_aux 的数，把"训练时每层多付多少"量出来。

        Returns:
            dispatch_weights: [N, k] fp32，被容量丢弃的 slot 是**精确的 0.0**；
            expert_indices:   [N, k] int64；
            aux_loss:         标量 tensor（可导，梯度只走 P_i）；compute_aux=False 时是 0；
            metrics:          dict（Python 标量，可直接进 JSON）。
        """
        n_tok = x.shape[0]
        E, k = self.num_experts, self.top_k

        # gate 全程 fp32：fp16 下 softmax 的 exp 容易在 logit 差 >10 时失去分辨率，
        # 而 gate 的算力占比 (<1%) 不值得为它省这点带宽。
        logits = x.float() @ self.weight.float()                      # [N, E]

        if self.training and self.jitter_noise > 0.0:
            # Switch 的 jitter：乘性噪声，均值仍为 1，不改变期望路由分布
            logits = logits * (1.0 + torch.empty_like(logits).uniform_(
                -self.jitter_noise, self.jitter_noise
            ))
        logits = logits + self.expert_bias

        probs = torch.softmax(logits, dim=-1)                          # [N, E] fp32
        top_p, top_i = torch.topk(probs, k, dim=-1)                    # [N, k]

        # 归一化 top-k（Mixtral 做法）：让每 token 的 dispatch 权重和为 1。
        # 对比 Switch top-1：那边权重就是 top_p 本身，和 < 1，输出被"置信度"整体缩放。
        dispatch = top_p / top_p.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        # ---- aux loss（Switch Transformer 式）----
        if compute_aux:
            # k-hot 计数：top_k>1 时一个 token 给 k 个 expert 各投一票，所以 sum_i f_i = k。
            k_hot = F.one_hot(top_i, E).float().sum(dim=1)             # [N, E]
            f = k_hot.mean(dim=0)            # [E] 被路由比例 —— argmax 来的，**不可导**
            P = probs.mean(dim=0)            # [E] 平均 gate 概率 —— softmax 来的，**可导**
            # detach() 在数值上是空操作（one_hot 本来就无梯度），但把意图写死：
            # 梯度只通过 P 流回 gate。k=1 时 perfect balance 下 aux 最小值为 1.0；
            # top_k>1 时最小值是 k（因为 f 的和是 k），读数字时别拿 1.0 当基准。
            aux_loss = E * (f.detach() * P).sum()
        else:
            aux_loss = logits.new_zeros(())

        # ---- 容量与丢弃 ----
        capacity = int(math.ceil((n_tok / E) * self.capacity_factor))
        rank = _position_in_expert(top_i, E)                           # [N, k] 队内序号
        if self.dropless:
            keep = torch.ones_like(rank, dtype=torch.bool)
        else:
            keep = rank < capacity
        dispatch = dispatch * keep.to(dispatch.dtype)   # 丢弃 slot 权重精确置 0

        # 只有需要统计（或 dropless 需要 max load 定 buffer 大小）时才算 load：
        # 那一次 bincount + max 也是要花 kernel 的。
        need_load = collect_metrics or self.dropless
        load = (
            torch.bincount(top_i.reshape(-1), minlength=E) if need_load else None
        )                                                              # [E] int64 分配数

        # dropless 的 buffer 大小必须等于"这一批里最热的 expert 收到多少条"，
        # 那是运行期才知道的量，躲不掉一次 D2H（.item()）。生产实现会改成
        # 固定桶 + 溢出处理或 kernel 内动态分段，eager PyTorch 里只能同步一次。
        # 有容量限制时容量是编译期算出来的整数，不需要同步。
        cap_used = capacity if not self.dropless else int(load.max().item())

        metrics: Dict[str, Any] = {
            "num_tokens": n_tok,
            "num_experts": E,
            "top_k": k,
            "dropless": self.dropless,
            "capacity_factor": self.capacity_factor,
            "capacity": capacity,
            # 层里开 slot buffer 用的实际容量
            "capacity_used": cap_used,
            "aux_loss": 0.0,
        }

        if collect_metrics:
            # 从这里开始每条都要一次 .item()，即一次 host 同步。
            n_dropped = int((~keep).sum().item())
            kept = n_tok * k - n_dropped
            dropped_tokens = int((~keep).any(dim=-1).sum().item())
            mean_load = float(load.float().mean().item())
            slots = max(E * max(cap_used, 1), 1)
            metrics.update(
                {
                    "num_dropped": n_dropped,
                    # 分配级丢率：分母是 slot 数
                    "drop_rate": n_dropped / (n_tok * k),
                    # token 级：至少丢了一个 slot 的 token 占比
                    "dropped_token_rate": dropped_tokens / n_tok,
                    # 有效吞吐：真正被计算的 token 占比（= 1 - token 级丢率）
                    "effective_token_ratio": 1.0 - dropped_tokens / n_tok,
                    # 负载不均衡度：最热 expert 的分配数 / 平均分配数。
                    # 均衡时 ~1.0；一个 expert 吃掉所有 token 时 ~E。这个数**在丢弃前**算，
                    # 因为它是路由本身的性质，与容量策略无关（容量只决定丢多少）。
                    "max_load_ratio": float(load.max().item()) / max(mean_load, 1e-9),
                    "min_expert_load": int(load.min().item()),
                    "max_expert_load": int(load.max().item()),
                    "kept_assignments": kept,
                    # 被"预定"的容量里真正填了多少：capacity_factor 越大这个越低，
                    # 就是"算力/显存白白预留"的那部分（实验 B 的 trade-off 右半边）。
                    "slot_utilization": kept / slots,
                    "wasted_slot_ratio": 1.0 - kept / slots,
                    # compute_aux=False 时这里没有意义，写 None 而不是假的 0.0
                    "aux_loss": float(aux_loss.item()) if compute_aux else None,
                    "expert_load": [int(v) for v in load.tolist()],
                }
            )
        # 不统计时 capacity_used 也已经算好（上面），layer 靠它开 buffer，不需要再同步。
        return dispatch, top_i, aux_loss, metrics