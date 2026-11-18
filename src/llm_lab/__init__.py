"""llm-inference-lab: 四个推理关键路径的最小可运行实现。

模块边界刻意按"可独立验证"切分：
  flash_attn     计算密集 —— 一个 attention 头在 SRAM 里怎么不落回 HBM
  paged_attn     显存管理 —— KV cache 怎么按块分配、怎么被 prefix 复用
  moe            参数/计算解耦 —— 稀疏激活带来的路由与负载均衡问题
  spec_decode    串行解码的墙 —— 用便宜模型猜、用贵模型验

每个模块都提供两条路径：
  reference()  朴素 PyTorch 实现，用于数值对齐（correctness oracle）
  fast()       Triton / 分块实现，用于性能对比
"""

__version__ = "0.1.0"