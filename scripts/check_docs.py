#!/usr/bin/env python
"""文档数字一致性检查。

为什么需要它：这套文档有 5 份（README + 4 篇 docs + 报告 + 计划书），
同一条实验结论会被引用十几次。实验重跑一次，数字就会漂，
而"某一份里还留着旧数"是最容易被评审一眼看穿的地方——
本项目实际发生过 3 次（MoE 重跑、spec 端到端重跑、paged 加速比重测）。

检查两类问题：
  1. 禁用词：被推翻的旧数字不该再出现（白名单里注明允许的例外）
  2. 必含词：当前结论的关键数字必须至少出现在正文里

用法：
    python scripts/check_docs.py          # 有问题是 exit 1
"""
from __future__ import annotations

import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (禁用字面量, 允许出现的文件, 允许理由)
FORBIDDEN = [
    ("3.85×", [], "MoE batch=1 的旧值，已由 4.05× 取代"),
    ("1.05–1.18", [], "投机解码端到端的旧区间，已由 0.89–1.58 取代"),
    ("1.36–1.88", [], "投机解码重复测量散布的旧值，已由 1.44–1.85 取代"),
    ("0.978", [], "MoE N=1 的旧延迟"),
    ("2.011", [], "MoE N=1024 的旧延迟"),
    ("5.09×", [], "MoE batch=1 的另一版旧值"),
    ("0.244 ms", [], "paged decode 的旧延迟，已由 0.305 ms 取代"),
    ("18,208", [], "前缀命中省下的旧 token 数"),
    ("122 个", [], "旧 clamped 块数，已由 208 取代"),
]

# 允许在"说明历史"的上下文里出现（docs/02 §4.4 与 §3.3 明确在讲两次测量的差异）
ALLOW_CONTEXT = {
    "78.59": ["docs/02_paged_attention.md"],   # 只允许作为"修 bug 前"的对照值
    "4.22×": ["docs/02_paged_attention.md", "docs/05_面试问答.md"],
    "11.28×": ["docs/02_paged_attention.md", "docs/05_面试问答.md"],
}

REQUIRED = [
    ("129×", "FlashAttention 显存收益"),
    ("80.09%", "预分配浪费率"),
    ("0.89%", "分页浪费率"),
    ("26.5", "预分配并发序列"),
    ("133.1", "分页并发序列"),
    ("82.32%", "前缀命中率（大池）"),
    ("71.13%", "前缀命中率（128 块，修 bug 后）"),
    ("1.036 ms", "MoE batch=1 延迟"),
    ("4.05×", "MoE batch=1 相对等参数 dense"),
    ("1.13", "投机解码 k=5/k=1 前向比"),
    ("4.4%", "散文接受率"),
    ("91.5%", "代码接受率"),
    ("0.89–1.58", "投机解码端到端区间"),
]


def docs() -> list[str]:
    files = glob.glob(os.path.join(ROOT, "*.md"))
    for sub in ("docs", "reports", "proposals", "third_party"):
        files += glob.glob(os.path.join(ROOT, sub, "*.md"))
    return sorted(f for f in files if os.path.basename(f) != "README.md" or True)


def main() -> int:
    files = docs()
    if not files:
        print("找不到任何 .md")
        return 1

    problems = 0
    print(f"扫描 {len(files)} 份文档\n")

    for literal, allowed, reason in FORBIDDEN:
        for path in files:
            rel = os.path.relpath(path, ROOT).replace("\\", "/")
            with open(path, encoding="utf-8") as f:
                for lineno, line in enumerate(f, 1):
                    if literal in line:
                        print(f"[禁用] {rel}:{lineno}  含 '{literal}'  ({reason})")
                        print(f"        {line.strip()[:100]}")
                        problems += 1

    for literal, allowed in ALLOW_CONTEXT.items():
        for path in files:
            rel = os.path.relpath(path, ROOT).replace("\\", "/")
            if rel in allowed:
                continue
            with open(path, encoding="utf-8") as f:
                for lineno, line in enumerate(f, 1):
                    if literal in line:
                        print(f"[越界] {rel}:{lineno}  含 '{literal}'，但只允许出现在 {allowed}")
                        problems += 1

    all_text = ""
    for path in files:
        with open(path, encoding="utf-8") as f:
            all_text += f.read()

    print("必含项检查：")
    for literal, what in REQUIRED:
        ok = literal in all_text
        print(f"  {'ok  ' if ok else 'MISS'}  {literal:<12} {what}")
        if not ok:
            problems += 1

    print()
    if problems:
        print(f"发现 {problems} 个问题")
        return 1
    print("全部通过：没有残留旧数字，关键结论都在。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())