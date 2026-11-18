# 代码库与实验资源清单

> 这份清单的标准只有一条：**每个项目都写清"读它为了回答哪个问题"**，
> 而不是罗列 star 数。带 ★ 的是我认为投入产出比最高的四个入口。
> 许可证以各仓库当前 LICENSE 为准；我只在最确定的地方直接标注。

---

## A. 代码库（按与本仓库四个模块的对应关系排）

### ★ 1. Dao-AILab/flash-attention — FlashAttention 官方实现
- **是什么**：FA1/FA2/FA3 的 CUDA 实现，含 Hopper 专用路径与 Python 接口。
- **读它回答什么问题**：我的 Triton 版在 tile 选择、mask 处理、`num_stages` 上与官方差在哪；
  以及为什么 FA3 在 Ada 上用不了（看 `hopper/` 目录里对 TMA / WGMMA 的依赖）。
- **建议入口**：先看 Python 侧的 `flash_attn/` 接口签名（`causal`、`window_size`、
  `softmax_scale`、varlen 参数），再进 `csrc/`。
- **许可证**：BSD-3-Clause。
- **对应本仓库**：`src/llm_lab/flash_attn/`。

### ★ 2. vllm-project/vllm — 生产级推理引擎
- **是什么**：PagedAttention 的诞生地，也是目前最完整的推理系统参考实现。
- **读它回答什么问题**：块池与引用计数怎么组织、前缀哈希的 key 怎么拼、
  一次 `allocate_slots` 怎么同时处理"命中 + 新块 + 配额"、投机解码的验证循环长什么样。
- **建议入口**（0.32 的 v1 架构）：`vllm/v1/core/block_pool.py` →
  `vllm/v1/core/kv_cache_manager.py` → `vllm/v1/core/single_type_kv_cache_manager.py`。
  旧版本叫 `vllm/core/prefix_caching_block.py`，网上大量博客还在用旧路径。
- **许可证**：Apache-2.0。
- **对应本仓库**：`src/llm_lab/paged_attn/`，逐文件对照见 `third_party/README.md`。

### ★ 3. triton-lang/triton — 写 kernel 的地基
- **是什么**：GPU kernel 的 Python DSL 与编译器。
- **读它回答什么问题**：`tl.dot` 的形状约束、block pointer / tensor descriptor 的用法、
  自动调优怎么配。**`python/tutorials/06-fused-attention.py` 是全世界最好的
  FlashAttention 入门材料**——比任何博客都直接。
- **建议入口**：`python/tutorials/` 里的 03-matrix-multiplication → 06-fused-attention
  → 09-persistent-matmul，按这个顺序读。
- **许可证**：MIT。
- **注意**：Windows 原生没有官方 wheel，需要社区版 `triton-windows`（本项目未验证）。

### 4. sgl-project/sglang — RadixAttention
- **是什么**：用**基数树**管理前缀复用，而不是 vLLM 的块级哈希链。
- **读它回答什么问题**：前缀复用的另一种数据结构取舍——树能表达"部分匹配"，
  块哈希只能表达"整块对齐的匹配"。这也是我 doc 02 §3.2 里"命中率上不去"的另一种解法。
- **许可证**：Apache-2.0。

### 5. NVIDIA/Megatron-LM — MoE 训练侧权威
- **是什么**：大规模训练框架，MoE 的路由、容量、并行策略都在这里。
- **读它回答什么问题**：grouped GEMM 在真实实现里怎么切、expert 并行与
  all-to-all 的通信模式、容量因子在训练里怎么设。
- **建议入口**：`megatron/core/transformer/moe/`（以仓库最新结构为准）。
- **对应本仓库**：`src/llm_lab/moe/`（我只做了单卡推理侧）。

### 6. microsoft/DeepSpeed — DeepSpeed-MoE
- **是什么**：MoE 的训练与推理支持，含 expert 并行与稀疏 kernel。
- **读它回答什么问题**：`capacity factor`、`top-k gating`、`dropless` 在工程上怎么落地。
- **建议入口**：`deepspeed/moe/`。
- **许可证**：Apache-2.0。

### 7. stanford-futuredata/megablocks — MoE 的 block-sparse kernel
- **是什么**：用 block-sparse 矩阵乘（而非 pad + bmm）实现 MoE。
- **读它回答什么问题**：**本仓库 doc 03 §2.2 里那 71.4% 零 pad 浪费的正解**。
  它把"不同专家拿到不同数量 token"这件事直接交给稀疏 kernel，而不是靠 padding 对齐。
- **许可证**：Apache-2.0。
- **这是我最想补上但没做的一块。**

### 8. LMCache/LMCache — KV cache 的跨请求复用
- **是什么**：把 KV cache 在实例间/存储层复用与 offload。
- **读它回答什么问题**：前缀缓存只在单实例单卡内有效，跨实例怎么办；
  KV 的序列化与传输成本怎么算。
- **许可证**：以仓库 LICENSE 为准。

### 9. NVIDIA/cutlass — 生产级 GEMM / FMHA
- **是什么**：CUDA 高性能矩阵乘与 fused attention 的模板库。
- **读它回答什么问题**：grouped GEMM 的真实实现、FMHA 的 warp 分工。
- **许可证**：BSD-3-Clause。
- **读它的前提**：先把 Triton 版本的 attention 写通，否则会被模板元编程淹没。

### 10. NVIDIA/TensorRT-LLM — 另一个生产实现
- **是什么**：含 paged KV、in-flight batching、投机解码的完整引擎。
- **读它回答什么问题**：同样的问题在 C++/CUDA 栈上怎么解，与 vLLM 的取舍差异。
- **许可证**：Apache-2.0。

### 11–12. 投机解码的两个代表实现
- **SafeAILab/EAGLE**：在**特征层**（倒数第二层）做自回归草稿，而不是 token 层。
  这是把接受率从 0.6 拉到 0.8 的关键动作，也是 doc 04 §6 里"EAGLE 2.7–3.5×"的来源。
- **FasterDecoding/Medusa**：多个轻量解码头 + 树状注意力。
  读它主要是看**树 mask 怎么构造**——我在 `spec_decode/tree.py` 里做的就是这件事。
- 两者都是研究代码，许可证以仓库为准；生产可用性不如 vLLM 内置的 spec decode。

---

## B. 实验与 profiling 资源

### B.1 计时（本仓库的做法，也是最容易做错的一环）
```python
# 错：kernel 是异步的，墙钟把 launch 算进计算时间
t0 = time.time(); kernel(); t1 = time.time()

# 对：CUDA event + warmup + 取中位数
torch.cuda.synchronize()
start, end = torch.cuda.Event(True), torch.cuda.Event(True)
start.record(); kernel(); end.record()
torch.cuda.synchronize(); ms = start.elapsed_time(end)
```
配套三件事缺一不可：**warmup**（进稳定时钟）、**取中位数**（抗离群）、
**记录 min/p90**（判断离散程度）。见 `src/llm_lab/common.py`。

### B.2 查笔记本节流（我的数字飘就是因为它）
```bash
nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu,utilization.gpu \
           --format=csv -l 1
```
RTX 4050 Laptop 在 6 GB 满载时会在 60–85 W 之间来回，SM 频率能差 30%。
**任何端到端百分比在笔记本上都要先怀疑节流**，而不是先怀疑代码。

### B.3 Nsight Compute (ncu) / Nsight Systems (nsys)
- `ncu` 回答"这个 kernel 是 memory-bound 还是 compute-bound"：
  看 `Memory Throughput %`、`Compute (SM) Throughput %`、`Achieved Occupancy`、
  `Registers Per Thread`（寄存器溢出会直接砍 occupancy）。
- `nsys` 回答"时间花在哪个 kernel/哪个 gap 上"：
  本仓库 MoE 那 48 个 kernel 的 launch 间隙，用 nsys 一眼就能看出来。
- 用法要点：先 `--set full` 扫一遍找热点，再对单个 kernel 做 `--section MemoryWorkloadAnalysis`。
  对很小的 kernel，ncu 的重放开销会让结论失真，要用 `--replay-mode application`。

### B.4 Torch 侧
- `torch.profiler`：看算子级时间与调用次数（本仓库统计 kernel 数就是靠 hook + profiler）。
- `torch.cuda.reset_peak_memory_stats()` + `max_memory_allocated()`：测单次调用的显存峰值增量。
- `torch.backends.cuda.flash_sdp_enabled()` / `mem_efficient_sdp_enabled()`：
  确认 SDPA 到底派发到了哪个后端，否则"我比 SDPA 快"可能比的是 math 后端。

---

## C. 负载与数据集（做仿真要用真实长度分布）

| 资源 | 用来做什么 | 备注 |
|---|---|---|
| ShareGPT / LMSYS-Chat-1M | 真实对话的输入/输出长度分布 | 碎片与命中率仿真的首选；vLLM 论文用的就是 ShareGPT |
| LongBench、RULER | 长上下文能力与 attention 成本 | 用来确认"长上下文下 FA 的收益" |
| HumanEval / MBPP | 投机解码收益最明显的场景 | 代码文本可预测性高，接受率能到 90%+ |
| 自造结构化文本 | 前缀命中率实验 | 本仓库用 lognormal 长度 + 共享 system prompt 构造，见 `paged_attn/bench.py` |

**一个提醒**：仿真用的长度分布决定了所有结论。我用 lognormal(mean≈600) 得到
"预分配浪费 80%"，换成全部等长 512 的负载，这个数字会掉到 0——
所以 doc 02 的结论只对"长度有长尾"的负载成立，我在"有效性威胁"里写了这一条。

---

## D. 如果要从头学一遍，我会这样排

| 顺序 | 做什么 | 验收标准 |
|---|---|---|
| 1 | 用 Triton 写一个 matmul，跑通并测 TFLOPs | 能解释为什么 tile 取 128×128 |
| 2 | 读 `triton/python/tutorials/06-fused-attention.py`，抄一遍，再脱稿写一遍 | 能说清 online softmax 三行递推 |
| 3 | 写 paged decode kernel（1 query 对 n key） | 能说清为什么 decode 是带宽瓶颈 |
| 4 | 读 vLLM 的 `block_pool.py` + `kv_cache_manager.py` | 能画出一次 `allocate_slots` 的状态迁移 |
| 5 | 自己实现块表 + 链式哈希前缀缓存，跑命中率实验 | 能解释"为什么池子再大命中率也不涨" |
| 6 | 实现拒绝采样验证，用 χ² 证明分布不变 | 能说清残差分布为什么能还原 p |
| 7 | 读 megablocks，理解 grouped GEMM 怎么消掉 padding | 能算出 pad 方案浪费了多少算力 |
| 8 | 用 ncu 对第 2 步的 kernel 做一次 roofline 定位 | 能指着图说"这里 memory-bound" |

每一步都要有**能给别人看的产物**（脚本 / 图 / 数字）。只看不写，等于没学。

---

## E. 怎么判断一个"权威代码库"值不值得读

三个筛子，按顺序用：

1. **有没有对应的论文，且论文里报的数字能被复现**（连数字都不敢报的项目，通常有问题）。
2. **README 里有没有"已知限制"或"not supported"**（只讲优点的项目要警惕，
   本仓库自己就在 README 里列了六条局限）。
3. **测试目录的密度**。`tests/` 比 `src/` 还厚的项目，通常是真在生产里用过的；
   一个 kernel 项目如果没有数值对齐测试，它的"快"很可能来自算错了。

按这三条筛，上面 A 里前 7 个都过关，11–12 属于"论文很好、工程一般"的研究代码。