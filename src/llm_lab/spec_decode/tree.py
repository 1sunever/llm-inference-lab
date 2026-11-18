"""树状草稿 + tree attention mask。

为什么需要树
--------------------------------------------------------------------------------
线性草稿（Speculative Decoding, arXiv:2211.17192）每轮只验一条链：k 个草稿
token 被拒掉一个，后面的全废。EAGLE / Medusa 的改进是把多条候选组织成一棵树，
一次前向验证所有分支：

        root(已验过的最后一个 token)
          |
          a
         / \\
        b   c
       / \\   \\
      d   e   f

关键观察：**多条候选链的公共前缀只算一次**。上面 6 个位置（不含 root）如果按
3 条独立链做就是 9 个位置，合并成树只要 6 个。树越大，省下的重复计算越多，
但 attention 必须同时满足两件事：

  1. 因果性：位置 i 不能看见"未来"的位置（按树的后序/深度优先编号时，
     这个约束不能靠下三角矩阵自动满足）；
  2. 树约束：位置 i 只能看见它自己的祖先链 + 自己。兄弟分支、无关分支
     必须被 mask 掉，否则 b 会看到 c 的 KV，算出来的 logits 就是错的
     （比"多算"更糟：结果不可用）。

所以 mask 不是 torch.tril，而是"祖先闭包"矩阵：
    mask[i, j] = True  <=>  j 是 i 的祖先或 j == i
跑一遍 `_build_ancestor_closure` 就能得到 T×T 的 bool 矩阵。

位置编号的约定（与 Medusa/EAGLE 的实现一致）：
  * 索引 0 是 root，代表"上下文里已经验证过的最后一个 token"；
  * 其余位置按"父节点一定排在子节点前面"（拓扑序）编号，这样 mask 可以原地构造；
  * `depth` 同时充当归一化位置编号（RoPE 的 position_ids 用 depth 偏移，
    见 EAGLE 的 "position ids from tree depth"），所以这里一并返回。
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch

__all__ = [
    "build_tree",
    "tree_attention_mask",
    "tree_paths",
    "ancestors",
    "tree_stats",
    "test_tree_mask",
]


def build_tree(
    draft_tokens_list: Sequence[Sequence[int]], root_token: int = 0
) -> Tuple[List[int], List[int], List[int]]:
    """把多条候选草稿合并成一棵树（重复前缀只留一份）。

    参数
    ----
    draft_tokens_list  若干条候选链，每条是 token id 列表；长度可以不同
    root_token         根节点的 token id（真实实现里是"已验过的最后一个 token"，
                       这里默认 0 只作为占位，mask/结构不依赖它的值）

    返回
    ----
    (parent_indices, token_ids, depth)
        parent_indices[i]  父节点下标；根为 -1
        token_ids[i]       该位置要喂给模型的 token id
        depth[i]           深度（根为 0），也是该位置相对 root 的"位置偏移"

    合并逻辑就是一棵 trie：从根出发，沿途遇到已存在的 (父, token) 边就复用，
    否则新建节点。因此同一条链重复传入不会产生新节点。

    例：[[10,20,30], [10,20,40], [10,50]] ->
        idx : 0    1   2   3   4   5
        tok : 0   10  20  30  40  50
        par : -1   0   1   2   2   1
        dep : 0    1   2   3   3   2
    """
    parent_indices: List[int] = [-1]
    token_ids: List[int] = [int(root_token)]
    depth: List[int] = [0]
    children: Dict[Tuple[int, int], int] = {}

    for chain in draft_tokens_list:
        cur = 0
        for tok in chain:
            key = (cur, int(tok))
            nxt = children.get(key)
            if nxt is None:
                nxt = len(parent_indices)
                children[key] = nxt
                parent_indices.append(cur)
                token_ids.append(int(tok))
                depth.append(depth[cur] + 1)
            cur = nxt
    return parent_indices, token_ids, depth


def ancestors(parent_indices: Sequence[int], idx: int) -> List[int]:
    """idx 的祖先链（从根到父节点，不含自己）。"""
    out: List[int] = []
    j = parent_indices[idx]
    while j >= 0:
        out.append(int(j))
        j = parent_indices[j]
    out.reverse()
    return out


def tree_attention_mask(parent_indices: Sequence[int], depth: Sequence[int]) -> torch.Tensor:
    """构造因果 + 树约束的 bool mask，形状 [1, 1, T, T]，True = 可见。

    mask[i, j] = True 当且仅当 j 在 i 的祖先链上或 j == i。
    这个矩阵天然包含因果性（祖先一定排在前面，所以是某种下三角，
    但比 tril 更稀疏：兄弟分支互相不可见）。

    实现是 O(T·depth) 的"沿父指针上溯"，T 是树大小（一般 <64），
    相比一次前向的开销可以忽略；不需要 CUDA kernel。

    返回 dtype=torch.bool。注意 PyTorch 的 SDPA 里 attn_mask 是 True=保留，
    flash-attn 的 cu_seqlens/块对角形式语义不同，转换时别搞反。
    """
    T = len(parent_indices)
    if len(depth) != T:
        raise ValueError(f"parent_indices 与 depth 长度不一致: {T} vs {len(depth)}")
    if T == 0:
        raise ValueError("树不能为空")
    if parent_indices[0] != -1:
        raise ValueError("索引 0 必须是根，parent_indices[0] 应为 -1")
    if depth[0] != 0:
        raise ValueError("根的 depth 必须是 0")

    mask = torch.zeros(T, T, dtype=torch.bool)
    for i in range(T):
        p = int(parent_indices[i])
        if p >= i:
            raise ValueError(
                f"节点 {i} 的父节点 {p} 必须小于它（要求拓扑序），否则 mask 无法原地构造"
            )
        if p >= 0 and depth[i] != depth[p] + 1:
            raise ValueError(f"depth 不自洽: depth[{i}]={depth[i]} 应为 depth[{p}]+1")
        mask[i, i] = True
        j = p
        while j >= 0:
            mask[i, j] = True
            j = int(parent_indices[j])
    return mask.view(1, 1, T, T)


def tree_paths(
    parent_indices: Sequence[int], token_ids: Sequence[int], depth: Sequence[int]
) -> List[List[int]]:
    """从树里恢复出所有"叶子到根"的候选链（用于核对合并前有哪几条草稿）。

    返回每条链的 token id 列表（不含根），按叶子下标排序。
    """
    T = len(parent_indices)
    has_child = [False] * T
    for i in range(1, T):
        has_child[int(parent_indices[i])] = True
    paths: List[List[int]] = []
    for i in range(T):
        if has_child[i]:
            continue
        chain: List[int] = []
        j = i
        while j > 0:
            chain.append(int(token_ids[j]))
            j = int(parent_indices[j])
        chain.reverse()
        paths.append(chain)
    return paths


def tree_stats(
    parent_indices: Sequence[int], token_ids: Sequence[int], depth: Sequence[int]
) -> dict:
    """树的统计量：节点数 / 最大深度 / 叶子数 / 合并省下的位置数。"""
    T = len(parent_indices)
    leaves = sum(1 for i in range(T) if not any(int(p) == i for p in parent_indices))
    max_depth = max(depth) if depth else 0
    # 若把每条根->叶路径单独跑一遍（每条链都要带上根），需要的总位置数
    paths = tree_paths(parent_indices, token_ids, depth)
    separate = sum(len(p) + 1 for p in paths)
    return {
        "num_nodes": T,
        "num_draft_positions": T - 1,
        "max_depth": int(max_depth),
        "num_leaves": int(leaves),
        "positions_if_separate": int(separate),
        "saved_positions": int(separate - T),
        "saved_ratio": round(1.0 - T / separate, 4) if separate else 0.0,
    }


# --------------------------------------------------------------------------- #
# 正确性测试：对一棵具体的小树逐元素断言 mask
# --------------------------------------------------------------------------- #


def test_tree_mask(verbose: bool = True) -> bool:
    """断言"非祖先位置被 mask 掉、祖先链可见"，通过则打印 PASS。

    用的树（3 条候选链共享前缀 10,20）：
        chains = [[10,20,30], [10,20,40], [10,50]]
        idx : 0=root  1=10  2=20  3=30  4=40  5=50
        parent: [-1, 0, 1, 2, 2, 1]   depth: [0,1,2,3,3,2]
    逐条检查：
      * 30(idx3) 能看到 {0,1,2,3}（自己的祖先链 + 自己），看不到兄弟 40(idx4)、
        看不到叔节点 50(idx5)；
      * 40(idx4) 能看到 {0,1,2,4}，看不到 30/50；
      * 50(idx5) 能看到 {0,1,5}，看不到 20 的两个子节点 30/40（它是 20 的兄弟）；
      * root 看不到任何子节点（只有自己可见）；
      * 每一行可见数 == depth + 1（这条性质对任意树都成立，是最强的单点断言）。
    """
    chains = [[10, 20, 30], [10, 20, 40], [10, 50]]
    parent, tokens, depth = build_tree(chains)

    assert parent == [-1, 0, 1, 2, 2, 1], parent
    assert tokens == [0, 10, 20, 30, 40, 50], tokens
    assert depth == [0, 1, 2, 3, 3, 2], depth

    mask = tree_attention_mask(parent, depth)
    assert mask.dtype == torch.bool, mask.dtype
    assert tuple(mask.shape) == (1, 1, 6, 6), tuple(mask.shape)

    m = mask[0, 0]

    # 1) 可见集合必须精确等于 {自己} ∪ 祖先
    for i in range(len(parent)):
        visible = set(m[i].nonzero(as_tuple=True)[0].tolist())
        expect = set(ancestors(parent, i)) | {i}
        assert visible == expect, f"节点 {i} 可见集合 {visible} != 期望 {expect}"

    # 2) 明确列出"非祖先位置被 mask 掉"的几对
    must_be_masked = [
        (3, 4),  # 兄弟：30 不能看见 40
        (4, 3),  # 兄弟：40 不能看见 30
        (3, 5),  # 堂兄弟：30 不能看见 50
        (5, 3),  # 堂兄弟：50 不能看见 30
        (5, 2),  # 50 是 20 的兄弟，不能看见 20
        (2, 5),  # 反向也不行
        (0, 1),  # 根不能看见子节点
        (1, 3),  # 10 是 30 的祖父，但"祖先"方向是子看父，父不看子
        (2, 4),  # 20 不能看见子节点 40
    ]
    for i, j in must_be_masked:
        assert not bool(m[i, j]), f"节点 {i} 不应看见 {j}（非祖先位置没被 mask 掉）"

    # 3) 明确列出"必须可见"的几对
    must_be_visible = [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0),  # 根对所有节点可见
        (3, 1), (3, 2), (3, 3),  # 30 的祖先链 + 自己
        (4, 4), (5, 5),
        (5, 1),  # 50 的父节点
    ]
    for i, j in must_be_visible:
        assert bool(m[i, j]), f"节点 {i} 应该看见 {j}"

    # 4) 行和性质：任意树、任意节点都满足 |visible(i)| == depth(i) + 1
    for i in range(len(parent)):
        assert int(m[i].sum()) == depth[i] + 1, (
            f"节点 {i} 可见数 {int(m[i].sum())} != depth+1 = {depth[i] + 1}"
        )

    # 5) 重复前缀必须被合并：同一条链传两次不产生新节点
    p2, t2, d2 = build_tree([[7, 8, 9], [7, 8, 9]])
    assert len(p2) == 4, f"重复链没有被合并，得到 {len(p2)} 个节点"
    assert t2 == [0, 7, 8, 9], t2

    # 6) 树本身也是合法的：拓扑序、depth 单调
    for i in range(1, len(parent)):
        assert parent[i] < i
        assert depth[i] == depth[parent[i]] + 1

    # 7) 分叉处的"公共前缀只留一份"：3 条独立链（每条带根）本应是 4+4+3=11 个位置，
    #    合并成树只要 6 个位置
    stats = tree_stats(parent, tokens, depth)
    assert stats["positions_if_separate"] == 11, stats
    assert stats["num_draft_positions"] == 5, stats
    assert stats["saved_positions"] == 5, stats
    assert sorted(tree_paths(parent, tokens, depth)) == [[10, 20, 30], [10, 20, 40], [10, 50]]

    if verbose:
        print("tree attention mask test: PASS "
              f"(T={len(parent)}, branches={len(chains)}, saved_positions={stats['saved_positions']})")
    return True


if __name__ == "__main__":
    test_tree_mask(verbose=True)