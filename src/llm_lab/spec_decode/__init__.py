"""spec_decode: 投机解码（draft -> verify）的最小可运行实现。

模块切分与流水线顺序一致，每一层都能单独验证：

  sampling.py  验证与采样。拒绝采样 + 残差修正，输出分布与直接对目标采样
               逐 token 精确相同（Leviathan et al. 2211.17192 / Chen et al.
               2302.01318）。加速比与最优 k 的解析公式也在这里。
  tree.py      树状草稿的合并与 tree attention mask（EAGLE / Medusa 的核心
               数据结构）：多条候选共享前缀，一次前向验证所有分支。
  draft.py     不需要草稿模型的 prompt-lookup / n-gram 起草器。
  bench.py     三组实验：分布等价性（正确性）、α 与加速比（理论 vs 蒙特卡洛）、
               真实工作负载上的接受率与端到端吞吐。

典型的用法：

    from llm_lab.spec_decode import PromptLookupDrafter, speculative_verify
    draft = PromptLookupDrafter(max_ngram=3, max_draft=5).forward(context_ids)
    result = speculative_verify(q_probs, p_probs, draft)   # q_probs 是 one-hot
    output.extend(result.tokens)

设计原则：正确性（分布等价）优先于速度；所有解析公式都与蒙特卡洛模拟对照，
不靠"看起来合理"的数字支撑结论。
"""

from llm_lab.spec_decode.draft import PromptLookupDrafter, longest_common_prefix_len
from llm_lab.spec_decode.sampling import (
    VerifyResult,
    distribution_distance,
    expected_accepted_tokens,
    expected_acceptance_rate,
    expected_speedup,
    naive_accept_draft,
    naive_verify_no_correction_batch,
    optimal_k,
    speculative_verify,
    speculative_verify_batch,
)
from llm_lab.spec_decode.tree import (
    ancestors,
    build_tree,
    test_tree_mask,
    tree_attention_mask,
    tree_paths,
    tree_stats,
)

__all__ = [
    "PromptLookupDrafter",
    "longest_common_prefix_len",
    "VerifyResult",
    "speculative_verify",
    "speculative_verify_batch",
    "naive_accept_draft",
    "naive_verify_no_correction_batch",
    "expected_accepted_tokens",
    "expected_speedup",
    "optimal_k",
    "expected_acceptance_rate",
    "distribution_distance",
    "build_tree",
    "tree_attention_mask",
    "tree_paths",
    "ancestors",
    "tree_stats",
    "test_tree_mask",
]