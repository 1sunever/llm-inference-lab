# 第三方许可证原件与来源

这个目录里是**上游项目许可证的原文**，内容未做任何修改。

为什么保留它们：`third_party/README.md` 会把读者指向上游的具体文件
（例如 vLLM 的 `vllm/v1/core/block_pool.py`）。既然本仓库建立在这种对照之上，
"我参考的是哪个项目、它是什么许可、我读的是哪一版"就应该可追溯——
而不是只在正文里写一句"参考了 vLLM"。**本仓库没有拷贝它们的任何代码**
（原因与取舍见 [../README.md](../README.md)）。

抓取方式：GitHub REST API `GET /repos/{owner}/{repo}/license`，
抓取日期 **2026-11-18**。

| 项目 | SPDX | 文件 | 许可证 blob sha |
|---|---|---|---|
| [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) | BSD-3-Clause | `flash-attention.LICENSE` | `5860e4b33f…` |
| [vllm-project/vllm](https://github.com/vllm-project/vllm) | Apache-2.0 | `vllm.LICENSE` | `261eeb9e9f…` |
| [triton-lang/triton](https://github.com/triton-lang/triton) | MIT | `triton.LICENSE` | `1d0238e86b…` |
| [pytorch/pytorch](https://github.com/pytorch/pytorch) | NOASSERTION¹ | `pytorch.LICENSE` | `c23172f7af…` |

¹ GitHub 的自动识别对 PyTorch 的许可证文件返回 `NOASSERTION`：它的正文以
`From PyTorch:` 开头，是一段再分发声明 + BSD-3 风格条款，不是标准模板。
原文已按抓取内容保存，请以文件本身为准。

---

## 两点需要说清楚的边界

**1. `sha` 是 LICENSE 文件的 blob sha，不是仓库 commit。**
它能唯一标识"我抓的是哪一版许可证文本"，但不能说明我读的是哪一版代码。
后者在 `third_party/README.md` 里按 vLLM 0.32.0 说明。

**2. 保留许可证 ≠ 拥有版权。**
本仓库的 MIT 许可证（[../LICENSE](../LICENSE)）只覆盖本仓库自己的代码，
不覆盖这些上游项目的任何部分。如果将来有人把上游代码拷贝进本仓库，
必须同时遵守对应许可证（Apache-2.0 还要求保留 NOTICE 与修改声明）。

## 如果要真的 vendor 上游代码

正确的做法不是把文件拷进来，而是：

```bash
bash scripts/fetch_upstream.sh      # 拉到 _upstream/（已 gitignore）
```

然后按 `third_party/README.md` 的对照表去读。这样既保持许可证清晰，
也让"哪部分是我写的"不含糊——这一点在面试被追问时比什么都重要。