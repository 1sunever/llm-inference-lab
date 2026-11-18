"""不需要草稿模型的起草器：prompt-lookup / n-gram 匹配。

想法很朴素：如果"接下来要说什么"在**已经出现的上下文里**出现过，那么把
上一次出现之后的那些 token 拿来当草稿，成本是一次字符串/张量查找，比跑
一个 draft model 便宜几个数量级。这就是 vLLM 的 `--speculative-model
[ngram]`、llama.cpp 的 lookup decoding、以及各类 RAG/摘要/代码编辑场景里
最常见的"免费加速"来源。

适用边界（面试常问"这东西什么时候没用"）：
  * 有用：复制粘贴式输出（改代码、写测试、格式转换、日志/表格续写）、
    长文档问答（答案句子常在 prompt 里）、agent 反复输出同一段 JSON schema；
  * 没用：开放式创作、闲聊、采样温度高的情况。此时命中率接近 0，
    草稿全是空的，系统退化成普通自回归（正确性不受影响，只是没加速）。

与拒绝采样验证的关系
--------------------------------------------------------------------------------
本模块只负责"猜"。猜出来的 token 必须再过一遍 `sampling.speculative_verify`，
用目标模型 p 做接受/拒绝 + 残差修正，输出分布才不变。确定性起草器在这里相当于
把草稿分布取成 one-hot（q(x)=1）：于是接受概率 = min(1, p(x)/1) = p(x)，
被拒时残差分布 = norm(max(0, p - onehot))，依然满足定理前提。
"确定性草稿 = 退化分布"这个视角是很多人第一次实现时漏掉的点——他们以为
prompt lookup 不涉及概率所以不用验证，那样输出就变成"上下文里出现过的 token
被系统性放大"，分布已经偏了。
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

__all__ = ["PromptLookupDrafter", "longest_common_prefix_len"]


def longest_common_prefix_len(a: Sequence[int], b: Sequence[int]) -> int:
    """两个 token 序列的最长公共前缀长度。

    这是"目标就是真实续写时，草稿会被接受多少个"的度量（实验 C 用它当 oracle
    接受数）：被接受的一定是连续前缀，第一个不匹配的位置之后全部作废。
    """
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


class PromptLookupDrafter:
    """从上下文里找最长匹配后缀，把它的后续 n-gram 当草稿。

    参数
    ----
    max_ngram   最多用多少个 token 的后缀去匹配（匹配越长，命中越准但越难命中）
    max_draft   一次最多起草多少个 token（= 投机解码里的 k 的上限）
    min_ngram   最短匹配长度。1 表示"最后一个 token 出现过就行"，噪声大；
                实际系统一般用 2~3。

    行为约定
    --------
    * `forward(input_ids)` 只读输入、不改状态（除了内部统计计数器），确定性：
      同样输入必得同样草稿，因此可以在 CPU 上跑、可以单元测试、可以复现。
    * 返回长度 <= max_draft；没有命中返回 []（调用方应处理"零草稿"退化路径）。
    * "取最近一次出现"而不是最早一次：语言模型有强烈的局部重复性，
      最近的匹配通常更接近接下来的内容。这是启发式，不影响正确性。
    """

    def __init__(
        self,
        max_ngram: int = 3,
        max_draft: int = 5,
        min_ngram: int = 1,
        lookback: int | None = None,
    ) -> None:
        self.max_ngram = int(max_ngram)
        self.max_draft = int(max_draft)
        self.min_ngram = int(min_ngram)
        self.lookback = None if lookback is None else int(lookback)
        if self.max_draft < 1:
            raise ValueError("max_draft 必须 >= 1")
        if not 1 <= self.min_ngram <= self.max_ngram:
            raise ValueError("需要 1 <= min_ngram <= max_ngram")
        if self.lookback is not None and self.lookback < 1:
            raise ValueError("lookback 必须 >= 1 或 None")
        self.stats: Dict[str, int] = {
            "calls": 0,
            "hits": 0,
            "draft_tokens": 0,
            "ngram_hits_1": 0,
            "ngram_hits_2": 0,
            "ngram_hits_3": 0,
        }

    # ---------------------------------------------------------------- 内部

    def _match_start(self, ids: List[int], n: int) -> int:
        """返回"上一次出现后缀 ids[-n:]"的起点之后的位置；没有返回 -1。

        只在 i <= len(ids) - n - 1 的范围里找，保证匹配段完全落在上下文内部、
        且不与当前后缀自身重叠（否则会匹配到自己，起草出永远正确的废话）。
        lookback 限制往回的窗口（真实系统也会设，避免长上下文里全量扫描的 O(n) 成本）。
        """
        suffix = ids[-n:]
        hi = len(ids) - n - 1
        lo = 0 if self.lookback is None else max(0, len(ids) - n - self.lookback)
        for start in range(hi, lo - 1, -1):
            if ids[start:start + n] == suffix:
                return start + n
        return -1

    # ---------------------------------------------------------------- 对外

    def forward_with_match(self, input_ids: Sequence[int], max_draft: int | None = None) -> Tuple[List[int], int]:
        """返回 (草稿 token 列表, 命中的 n-gram 长度)；未命中为 ([], 0)。"""
        ids = [int(t) for t in input_ids]
        md = self.max_draft if max_draft is None else int(max_draft)
        self.stats["calls"] += 1
        if md <= 0 or len(ids) <= self.min_ngram:
            return [], 0
        longest = min(self.max_ngram, len(ids) - 1)
        for n in range(longest, self.min_ngram - 1, -1):
            pos = self._match_start(ids, n)
            if pos < 0:
                continue
            cont = ids[pos:pos + md]
            if not cont:
                return [], 0
            self.stats["hits"] += 1
            self.stats["draft_tokens"] += len(cont)
            key = f"ngram_hits_{n}"
            self.stats[key] = self.stats.get(key, 0) + 1
            return cont, n
        return [], 0

    def forward(self, input_ids: Sequence[int], max_draft: int | None = None) -> List[int]:
        """起草：返回长度 <= max_draft 的草稿 token（可能是空列表）。"""
        return self.forward_with_match(input_ids, max_draft=max_draft)[0]

    __call__ = forward

    def reset_stats(self) -> None:
        for k in self.stats:
            self.stats[k] = 0

    def hit_rate(self) -> float:
        calls = self.stats["calls"]
        return self.stats["hits"] / calls if calls else 0.0