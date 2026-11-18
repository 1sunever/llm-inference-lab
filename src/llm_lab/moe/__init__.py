"""MoE（Mixture-of-Experts）：参数与计算解耦。

对外三条路：
  TopKRouter   gate + top-k + 容量丢弃 + 负载均衡 aux loss
  MoELayer     路由 → scatter → 一次 bmm 算 E 个 expert → 加权收回
  DenseFFN     等参数量对照（每 token 全激活），MoE 的"如果不稀疏会怎样"

本模块的每个数字都由 bench.py 在真实 GPU 上跑出来，写进 results/moe_*.json；
没有任何手填的常数。想复现：bash _run/moe_bench.sh
"""

from .layer import DenseFFN, MoELayer, SwiGLUExpert, dense_hidden_for_params
from .router import TopKRouter

__all__ = [
    "TopKRouter",
    "MoELayer",
    "SwiGLUExpert",
    "DenseFFN",
    "dense_hidden_for_params",
]