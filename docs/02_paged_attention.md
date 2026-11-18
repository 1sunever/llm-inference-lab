# PagedAttention：显存碎片的账，以及前缀复用的真实收益

> 结论先行：同样 2.9 GB 的 KV 预算，预分配方案能并发 **26.5** 条序列，
> 分页方案能并发 **133.1** 条（5.0×）。差距不在算法，在**碎片**：
> 浪费率 80.09% → 0.89%。这就是 PagedAttention 的真实价值，
> 它不是一个更快的 attention，它是一个**显存分配器**。

---

## 1. 问题：为什么预分配必然浪费

在 vLLM 之前，主流做法是给每条序列按 `max_len` 预留一段连续 KV：

```python
kv = torch.zeros(num_seqs, 2, max_len, num_heads, head_dim)  # 一次要满
```

三个浪费来源，全部无法避免：

1. **预留过多**：请求的实际长度事先不知道，只能按上界要；
2. **内部碎片**：长度一旦确定，末尾用不满的那截也退不回去；
3. **外部碎片**：不同序列长度不一，中间会留下用不上的空洞。

vLLM 论文 §1 的原话是：现有系统里 "only **20.4% – 38.2%** of the KV cache
memory is used to store the actual token states"，§3.1 又强调
"the actual effective memory in previous systems can be **as low as 20.4%**"。

> 顺带纠正一个常被误引的数字：网上流传的 "浪费 60%–80%" 和 "PagedAttention
> 浪费 <4%" 都出自 vLLM 官方博客（2023-06-20），**不在论文里**。
> 论文自己的表述是 "near-zero waste"。引用时该引哪句，最好心里有数。

---

## 2. 机制：页表 + 引用计数 + 链式哈希

### 2.1 块表

一条序列的 KV 从"一段连续显存"降级成"若干固定大小的块 + 一张页表"：

```
logical block:  0    1    2    3
                 |    |    |    |
block_table:  [ 7,  19,   3,  42 ]   -> 物理块 id
```

代价是每次 attention 多一次查表（`paged_decode.py` 里就是 `bid = BT[b, i]`），
收益是内部碎片只剩最后一块、且**长度增长不需要 realloc**。

### 2.2 引用计数与写时复制

两个请求共享同一段前缀时，它们的 block_table 指向同一批物理块，
只把 `ref_count` 加 1（`BlockManager.fork`）。任何一方要写入时才真正复制
（`copy_on_write`）。这让"共享"变成零拷贝操作，而不是"复制一份再说"。

### 2.3 链式哈希（这是最容易写错的地方）

前缀复用要判断"这两段 token 是不是同一个前缀"。只哈希块内 token 会撞车：

```
序列 A: [the, cat] [sat, on]      块0 哈希 h([the,cat])，块1 哈希 h([sat,on])
序列 B: [sat, on] [the, cat]      块1 会撞上 A 的块0  —— 错
```

正确做法是**链式**：`h_i = H(h_{i-1}, block_i 的 token)`，
块 i 的哈希里带着它所有祖先的信息，撞车就不可能。

```python
def _hash_tokens(self, prev_hash: int, tokens: Tuple[int, ...]) -> int:
    h = hashlib.blake2b(digest_size=8)
    h.update(prev_hash.to_bytes(8, "little"))
    h.update(struct.pack(f"{len(tokens)}i", *tokens))
    return int.from_bytes(h.digest(), "little")
```

两个细节：
- 只用 `hashlib` 而不是 Python 内置 `hash()`：后者每次进程启动加盐，不可复现。
- **只对写满的块建索引**：没满的块内容还会变，复用会拿到错的 KV。

---

## 3. 实测（本机 RTX 4050 6 GB）

### 3.1 碎片与并发能力

配置：256 条序列，长度 lognormal(mean≈600)，clip 到 [32, 4096]，`max_len=4096`，
`block_size=16`，Qwen2.5-1.5B 的 KV 开销 28 KB/token（28 层 × 2 KV 头 × 128 × 2 × fp16）。

| 方案 | 分配 token | 实耗 token | 浪费 | 2.9 GB 预算下并发序列 |
|---|---|---|---|---|
| 连续预分配 | 1,048,576 | 208,775 | **80.09%** | 26.5 |
| 分页 (bs=16) | 210,656 | 208,775 | **0.89%** | **133.1** |

**5.0× 的并发差距**。注意分页的分配量 210,656 恰好比实耗大一点点——
那正是 256 条序列各自的最后一块，平均浪费半块（8 token），与理论一致。

### 3.2 前缀命中率

工作负载：4 个会话并发，每个会话 4 轮，共享 1024 token 的 system prompt，
每轮追加 ~96 token 用户输入 + ~120 token 生成。

| 池大小 | 命中率 | 省下的 prefill token | 被挤掉的块 |
|---|---|---|---|
| 4096 块 | **82.32%** | 19,072 / 23,168 | 0 |
| 512 块 | 82.32% | 19,072 / 23,168 | 0 |
| 128 块 | **78.59%** | 18,208 / 23,168 | 122 |

两条可讲的观察：

1. **512 与 4096 完全一样**——说明这个工作负载的活跃前缀集只有 ~270 块，
   池子过了这个阈值再大就没用了。"加显存能提命中率"是错的，要加对地方。
2. **128 块时掉到 78.59%，且出现 122 个"装不下的块"**——此时命中率这个指标
   已经部分失真，因为请求本身都放不全。所以我把 `clamped_blocks` 和命中率
   一起记进 JSON，不单独看命中率一个数。

前缀共享的论文对照：vLLM §6.4 报 one-shot 前缀下 1.67×、5-shot（341 token 前缀）
下 3.58× 的吞吐提升（对 Orca Oracle）。**但那是 A100 上多请求服务的吞吐，
不是我这张卡上单请求的延迟**，两者不能互相印证，报告里分开列。

### 3.3 decode kernel：零拷贝值多少

decode 阶段是「1 个 query 对 n 个 key」，读数远多于算数，是纯带宽瓶颈。

| 上下文 | fused paged kernel | 批量 gather + SDPA | 加速 |
|---|---|---|---|
| 128 | 0.046 ms | 0.399 ms | 8.66× |
| 256 | 0.030 ms | 0.335 ms | 11.28× |
| 512 | 0.053 ms | 0.353 ms | 6.63× |
| 1024 | 0.101 ms | 0.585 ms | 5.77× |
| 2048 | 0.244 ms | 1.029 ms | 4.22× |

（B=8, H=8, D=64, block_size=16）

gather 路径要多做一次 `n×D` 的显存搬运——**等于把分页省下的搬运算又搬一遍**。
上下文越长，这次拷贝的占比越高，所以加速比从 11.3× 降到 4.2×。

### 3.4 一个反直觉的事实

PagedAttention 的 kernel 本身**比连续布局更慢**。vLLM 论文 §7.1 写得很直白：
分页使 attention kernel 延迟增加 "**20–26%** higher ... compared to the
highly-optimized FasterTransformer implementation"。

它仍然全面胜出，因为省下的显存让 batch 变大，而吞吐 ≈ batch × 单请求效率。
**单请求慢了 20%，并发多了 5 倍，净赚 4 倍**——这个 trade-off 要能一句话说清，
否则"你为什么用一个更慢的 kernel"就答不上来。

---

## 4. 实现里踩过的三个坑（都是静默错误）

### 4.1 `evict()` 死循环

第一版驱逐逻辑：把 `ref_count > 0`（不可驱逐）的条目**放回 lru**，
继续 `while` 找下一个。结果 `popitem` 拿到的还是它，循环永远不结束。

```python
# 错：原地放回，同一个 while 里再次被 popitem 拿到
if blk.ref_count != 0:
    self.lru[h] = None
    continue
```

修法：先攒进 `skipped`，**循环结束**再放回。这个 bug 只在池子真的不够用时触发，
所以前面几个实验（池子够大）全部正常通过——典型的"只在压力下暴露"。

### 4.2 参考实现自己算错了

我给 `paged_attention_reference` 写了 `permute(0, 2, 1, 3)`，
把 H 夹到了 block_size 中间，结果"参考值"和"分页 gather"一致（同一个错误），
反而和连续布局 SDPA 差了 1.87。**一个 bug 把另一个 bug 掩盖了。**

正确答案是不用 permute：`k_cache[ids]` 已经是 `(nb, bs, H, D)`，
直接 `reshape(-1, H, D)` 就是正确的 token 顺序。

教训写进 `tests/` 了：**参考实现必须有第三条独立路径交叉验证**。
现在 C 组实验同时比 fused / gather / contiguous 三条路径，全部在 3e-4 内一致。

### 4.3 串行释放导致假命中率

最初的命中率实验每个请求结束就 `free`，结果 4096/512/128 三种池子测出**完全相同**
的 85.6%——因为峰值占用只有一条请求，池子大小根本没参与。

改成 4 会话并发占用后才有区分度。**一个没有压力的实验，测出来的数是假的**，
这类问题不看数字本身看不出来，只能靠"这个结果合理吗"来发现。

---

## 5. 局限

- **`deque.remove` 是 O(n)**。`claim()` 要在空闲队列里摘出指定块，块数上千时
  每次 O(n)。生产实现应该用侵入式双向链表（vLLM 的 `FreeKVCacheBlockQueue`
  就是这个），我保留了 deque 是为了代码可读，注释里标了这一点。
- **单卡、无 swap/抢占**。真实引擎在显存不够时会把序列 swap 到 CPU 或
  recompute，本仓库只做 LRU 驱逐索引。
- **block_size 固定为 16**，没有做块大小的敏感性扫描（论文里 16 是常用值，
  更大的块减少页表长度但增加内部碎片）。
- **前缀哈希没有做跨请求的 token 级校验**：只信哈希。理论上 8 字节 blake2b
  碰撞概率可忽略，但严格实现会存一份 token 用于比对。