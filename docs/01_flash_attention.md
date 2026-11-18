# FlashAttention：为什么它快，以及本仓库的实现与实测

> 结论先行：在 6 GB 的 RTX 4050 上，FlashAttention 的决定性收益**不是速度，是显存**。
> 我的 Triton 实现在 N=4096 处与 PyTorch SDPA 打平（1.01×），但在 N=2048 处
> 显存占用是朴素实现的 **1/129**（4.1 MB vs 532 MB）。把这条说清楚，比声称
> "比官方快 3 倍"更难被问倒。

---

## 1. 朴素实现错在哪

标准 attention：

```
S = Q Kᵀ / √d      (N×N)
P = softmax(S)     (N×N)
O = P V            (N×d)
```

问题不在 FLOPs（那是 4N²d，绕不过去），在于 **S 和 P 要被写回 HBM 再读出来**：

| | 朴素 | FlashAttention |
|---|---|---|
| 中间量显存 | O(N²) | O(N) |
| HBM 读写 | O(N²) | O(N²/M)（M=block 大小） |
| 算术强度 | ≈1 FLOP/byte | ≈N FLOP/byte |
| 瓶颈 | 显存带宽 | 算力（N 大时） |

算术强度这一行是关键。RTX 4050 的 machine balance 约 20–30 FLOP/byte，
而朴素实现只有 ~1，**天生 memory-bound**，再快的 tensor core 也等不到数据。
FlashAttention 把强度抬到 ~N，N≥64 时就跨进 compute-bound 区，
这才是"长序列收益更大"的定量解释——而不是"因为它用了 SRAM"。

实测印证（B=2, H=8, D=64, causal）：

| N | 朴素 | SDPA | 我的 Triton | 显存比（朴素/Triton） |
|---|---|---|---|---|
| 256 | 0.253 ms | 0.052 ms | 0.081 ms | 19.5× |
| 512 | 0.862 ms | 0.075 ms | 0.100 ms | 35.2× |
| 1024 | 4.50 ms | 0.174 ms | 0.205 ms | 66.4× |
| 2048 | 18.35 ms | 0.526 ms | 0.613 ms | 129.0× |
| 4096 | OOM | 1.791 ms | 1.765 ms | — |

（数字来自 `results/flash_attn_bench.json`，本机实跑）

> 注意 4096 处朴素实现直接 OOM：fp32 的 S 要 2×8×4096²×4 B = 1.07 GB，
> 加上 mask 和 P 的副本，6 GB 卡放不下。**这就是 FlashAttention 存在的理由**，
> 比任何速度数字都直接。

---

## 2. 算法：online softmax 的三行递推

把 K/V 沿序列切成 `BLOCK_N` 宽的块，逐个块更新三个运行量：

```
m_i  : 到目前为止见过的最大 logit（数值稳定用）
l_i  : 到目前为止的 exp 和
acc  : 到目前为止的加权 V 累加
```

每来一个新块：

```
m_new = max(m_i, max(qk))           # 新的全局最大
alpha = exp(m_i - m_new)            # 旧的累加要按比例缩小
acc   = acc * alpha + P_block @ V_block
l_i   = l_i * alpha + sum(P_block)
m_i   = m_new
```

三行就够，但缺一不可：
- 少了 `alpha` 缩放，旧累加的权重就错了（相当于用了过时的 max）
- 最后统一除一次 `l_i`，而不是每步都除（少 `num_blocks` 次除法）

`acc` 的形状是 `BLOCK_M × HEAD_DIM`，与 N 无关 —— **中间量从不落 HBM 的全部含义就在这里**。

---

## 3. 本仓库的四个实现决策（附理由）

代码在 `src/llm_lab/flash_attn/triton_fa2.py`。

### 3.1 STAGE 分离：把"要不要 mask"做成编译期常量

FA1 全程都带三角 mask，等于多做一倍的元素级比较。FA2 把这个循环拆成两段：

- `STAGE=1`：**整块都在对角线以下**，不需要任何 mask
- `STAGE=2`：对角块，需要三角 mask
- `STAGE=3`：非因果全序列，只需边界 mask

`IS_CAUSAL` 是 `tl.constexpr`，所以 causal 与非 causal 编译出**两个不同的 kernel**，
没有运行期分支。这是我照 FA2 论文 §3.2 做的，收益在 causal 场景下可直接测到。

### 3.2 `EVEN_N`：尾部块的静默错误陷阱

如果 N 不是 `BLOCK_N` 的整数倍，尾部块会读到越界数据。`make_block_ptr` 的
`boundary_check` 只会补 0，而**补 0 的 key 会被当成"相似度为 0 的有效 key"**，
悄悄污染 softmax 分母。N=1000 时误差看起来还能忍，N=4097 时直接崩。

处理方式：宿侧算好 `EVEN_N = (N % BLOCK_N == 0)` 作为 constexpr 传入，
`False` 时走显式掩码分支（把 `>= N_CTX` 的列压成 -1e6），
`True` 时**完全跳过这个分支**，不付代价。

这条是我实际踩过的：第一版没有 `EVEN_N`，测试里 N=1000 是通过的（因为
1000 = 8×125，恰好... 其实不整除，但因为 mask 掩盖了部分错误），
直到加入 N=4097 才暴露。现在 `tests/test_flash_attn.py` 里专门放了
1000 和 4097 两个用例。

### 3.3 exp2 而不是 exp

`exp(x) = exp2(x · log2 e)`。把 `log2 e` 折进 `qk_scale`（`sm_scale * LOG2E`），
省掉一次逐元素乘法。N 大时可测，N 小时被 launch 开销淹没。

### 3.4 tile 与 num_stages 的真实约束

不是"调参调出来的"，是两条硬约束：

1. **shared memory**：Ada 每 SM 约 99 KB（101376 B）。K/V tile 要乘 `num_stages` 份。
   `D=256, BLOCK_N=64, stages=3` 实测要 212992 B，直接
   `OutOfResources: shared memory`。所以 D=256 的配置退到 `BLOCK_M=64, BLOCK_N=32, stages=2`。
2. **寄存器**：`acc` 是 `BLOCK_M × HEAD_DIM` 的 fp32，`BLOCK_M` 一大就把寄存器吃满，
   occupancy 掉下来反而更慢。

`_CONFIGS` 里每个 head_dim 的取值都是这两条约束下测出来的，不是抄的。

---

## 4. 实测：三条诚实的结论

1. **与 SDPA 同量级，不主张速度优势**（N=4096：1.765 vs 1.791 ms，1.01×）。
   短序列落后最多 37%（N=256：0.081 vs 0.052 ms），原因是
   launch 开销 + tile 利用率不足，N 小的时候 grid 只覆盖 2 个 block，
   20 个 SM 里大部分闲着。
   **但这条结论的精度有限**：完整重跑一次（机器负载不同）后，triton/sdpa 的比值
   从 0.63–1.01 变成 0.83–1.19，N=4096 causal 甚至反超到 1.19×。
   两次合并看，**该比值在 0.63–1.19 之间波动**，即"与 SDPA 同量级"。
   所以本项目不主张速度优势，只主张显存优势——后者是 129 倍，不在噪声范围内。
2. **打不过朴素实现的场景不存在**：N≥256 时全部 2.9×–40× 更快。
3. **显存是数量级的差距**，见 §1 的表。

达成算力：N=4096, D=64 时 19.5 TFLOP/s。做 roofline 对照时要注意：
论文里 FlashAttention-2 的 230 TFLOP/s 是 **A100 80GB SXM4** 的数（§4.1），
225 TFLOP/s / 72% MFU 是 8×A100 端到端（Table 1），
335 TFLOP/s 是 **H100** 的（§4.1 末，不是 A100）——
这三个数经常被混着引用，报告里我按来源分开写。

**FA3（arXiv:2407.08608）与本机无关**：它依赖 Hopper 的 TMA/WGMMA/warp
specialization，H100 SXM5 上 740 TFLOP/s；sm_89（Ada）没有这些指令，
用不了也不能拿来对比。（顺带：常被引作 FA3 的 2407.08691 其实是
*ElasticAST*，一篇音频 spectrogram transformer 论文。）

---

## 5. 局限（不写清楚就是埋雷）

- **只实现前向**。反向需要重算 S（recomputation）+ dQ/dK/dV 的原子累加，
  推理用不到，但训练要用，本仓库不做。
- **不支持 varlen / GQA / paged KV**。真实推理里 Q 有 H 个头、KV 只有 H/4 个，
  以及变长 batch，都需要额外处理。
- **`tl.make_block_ptr` 在 Triton 3.7 已标记 deprecated**（仍可用）。
  迁移到 `tl.make_tensor_descriptor` 需要把 kernel 签名改成宿主侧传
  `TensorDescriptor` 对象；我保留 block_ptr 是因为它把 tiling 讲得更直白，
  README 的"已知问题"里记了这条。
- **没有做 INT8/FP8**。Ada 支持 FP8 E4M3，真有性能空间，但精度与
  scale 的故事会盖过 attention 本身，留作后续。

---

## 6. 可能被追问的点

| 问题 | 一句话回答 | 支撑位置 |
|---|---|---|
| 为什么快？ | 中间矩阵不落 HBM，算术强度从 ~1 抬到 ~N | §1 表 |
| 为什么 causal 要分两段？ | 恒定的因果区不需要 mask，省一半元素级比较 | `_attn_fwd_inner` 的 STAGE |
| N 不整除怎么办？ | constexpr `EVEN_N`，False 才走掩码分支，不给快路径加开销 | §3.2 |
| 为什么不直接用 flash-attn 包？ | 编译成本高且不可控；能改 tile 比能 import 更能说明问题 | 文件头注释 |
| 为什么没比 SDPA 快？ | 短序列是 launch-bound；长序列已持平。真正的收益在显存 | §4 |
| 能上 A100 跑吗？ | 能，代码不依赖 sm_89 特性；但数字会完全不同，不要混用 | §4 环境说明 |