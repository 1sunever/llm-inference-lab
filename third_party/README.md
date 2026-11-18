# 第三方来源与引用

本仓库**没有拷贝任何第三方源码**。四个模块的实现是按论文与公开源码独立写的，
第三方项目的用途是"对照阅读"：确认我的理解与工业实现一致，以及找出差异在哪。

这样处理有两个理由：

1. **许可证层面干净**。把 vLLM 的源码片段搬进来需要连同 Apache-2.0 的
   版权头、修改声明一起维护，工作量不小且容易漏。
2. **被追问时站得住**。如果代码是拷贝的，面试官问"你为什么这么写"就只能答
   "上游这么写"。现在是自己的实现，每一处取舍都有理由可讲，包括踩过的坑
   （见 `docs/02_paged_attention.md` §4）。

想自己对照的话，跑 `scripts/fetch_upstream.sh` 会把上游拉进 `_upstream/`
（已 gitignore），然后按下面的表格去读对应文件。

---

## 1. 对照阅读表

上游版本：**vLLM 0.32.0**（本机 `pip show vllm` 实测），路径相对于
`site-packages/vllm/`。

| 本仓库 | 上游对应位置 | 读它为了确认什么 |
|---|---|---|
| `paged_attn/block_manager.py` | `v1/core/block_pool.py` | 物理块池、引用计数、空闲队列的真实数据结构（上游用侵入式双向链表而不是 deque） |
| `paged_attn/block_manager.py`（分配入口） | `v1/core/kv_cache_manager.py` | 一次 `allocate_slots` 里怎么把"前缀命中 + 新块 + 配额"三件事合并 |
| `paged_attn/prefix_cache.py` | `v1/core/block_pool.py`（哈希与驱逐部分） | 链式哈希的 key 组成、LRU 的维护方式、命中判定 |
| `paged_attn/cache.py` | `v1/core/single_type_kv_cache_manager.py` | 块表怎么落到真正的张量视图上 |
| `flash_attn/triton_fa2.py` | `model_executor/kernels/attention/`、`vllm_flash_attn/` | 上游怎么处理 varlen / GQA / paged KV 的组合，以及 tile 怎么选 |
| `moe/router.py`、`moe/layer.py` | `model_executor/models/` 下各 MoE 模型（如 Mixtral / DeepSeek 系）+ 融合 MoE 算子 | 容量因子与 dropless 的实际取舍、grouped GEMM 怎么合并 kernel |
| `spec_decode/sampling.py` | `v1/spec_decode/`、`v1/watermarking/spec_decode.py` | 拒绝采样在工业实现里怎么写、bonus token 的特殊处理 |
| `spec_decode/tree.py` | `model_executor/models/medusa.py`、`*_eagle*.py` | 树状注意力的 mask 怎么构造与复用 |

> 说明：`vllm/v1/` 是 0.32 的新架构。旧版本里这块叫
> `vllm/core/prefix_caching_block.py`，很多博客和面试题还在用旧路径，
> 对照时注意别找错文件。

---

## 2. 论文

数字引用的来源与"哪些数字不能用在本机结论里"，逐条写在
`docs/` 各篇的引用处。核心几篇：

| 主题 | 论文 | 备注 |
|---|---|---|
| FlashAttention-2 | arXiv:2307.08691 | 230 TFLOP/s 是 A100 80GB SXM4；335 TFLOP/s 是 H100（易误引） |
| FlashAttention-3 | **arXiv:2407.08608** | 常被误写成 2407.08691（那个 ID 是 ElasticAST，一篇音频论文） |
| PagedAttention / vLLM | arXiv:2309.06180 (SOSP'23) | "20.4%–38.2% 有效利用率"是论文原话；"60%–80% 浪费"和"<4%"出自官方博客而非论文 |
| Switch Transformer | arXiv:2101.03961 | 7× 是 step-time / 样本效率，不是 wall-clock 吞吐 |
| GShard | arXiv:2006.16668 | top-2 容量 O(N/E)，溢出走残差 |
| Mixtral 8x7B | arXiv:2401.04088 | 47B 常驻显存 / 13B 激活，显存 ∝ 47B 是论文 §3 明确的 |
| DeepSeekMoE | arXiv:2401.06066 | 细粒度 + 共享专家；16.4B 总 / 2.8B 激活 |
| 投机解码（原始） | arXiv:2211.17192、arXiv:2302.01318 | 输出分布不变性的两篇奠基工作 |
| Medusa | arXiv:2401.10774 | 树状注意力 |
| EAGLE | arXiv:2401.15077 | 特征层起草 |
| EAGLE-2 | **arXiv:2406.16858** | 常被误写成 2404.18911（那个 ID 是 Kangaroo） |

两处 arXiv 编号的纠错值得单独记一笔：`2407.08691` 和 `2404.18911` 在网上
被大量二手资料错引，直接抄进报告会在评审那里露怯。

---

## 3. 工具

| 项目 | 许可证 | 用途 |
|---|---|---|
| PyTorch | BSD-3-Clause | 张量、SDPA 基线 |
| Triton | MIT | 自写 kernel 的编译与 launch |
| NumPy / Matplotlib | BSD-3-Clause / PSF-like | 数据处理与作图 |