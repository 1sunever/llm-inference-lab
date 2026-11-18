#!/usr/bin/env python
"""Markdown -> PDF（无外部依赖，只需一个 Chromium 内核浏览器 + 可选 PyMuPDF）。

为什么不用 pandoc/wkhtmltopdf：这台机器上没有，而 Edge/Chrome 一定在。
为什么不用 Python 的 markdown 库：只需要一个很小的子集（标题/段落/表格/代码块/
列表/引用/行内强调），自己解析 100 行就够，还省掉一个依赖。

用法：
    python scripts/md2pdf.py 输入.md 输出.pdf [--title "标题"]

实现分三步：
    1. Markdown -> 单文件 HTML（CSS 内联，图片用相对路径，所以要在原目录渲染）
    2. Edge/Chrome headless 打印为 PDF
    3. PyMuPDF 补页码、写元数据（可选；有就做，没有就跳过）

排版约定（A4 / 20mm 页边距 / 正文 11pt 行高 1.9）是为了中文技术文档的可读性，
代码块和表格允许跨页，标题与其后正文不分页。
"""

from __future__ import annotations

import argparse
import html
import os
import re
import shutil
import subprocess
import sys
import tempfile

CSS = r"""
@page { size: A4; margin: 20mm 18mm 18mm 18mm; }
* { box-sizing: border-box; }
body {
  font-family: "SimSun", "Songti SC", serif;
  font-size: 11pt; line-height: 1.9; color: #1a1a1a;
  max-width: 100%; margin: 0; text-align: justify;
}
h1, h2, h3, h4 { font-family: "SimHei", "Heiti SC", sans-serif; color: #111; }
h1 {
  font-size: 19pt; margin: 0 0 6mm 0; padding-bottom: 3mm;
  border-bottom: 2.2pt solid #1f3864; letter-spacing: .5pt;
  page-break-after: avoid;
}
h2 {
  font-size: 14pt; margin: 8mm 0 3mm 0; padding-left: 3mm;
  border-left: 4pt solid #2e5c9a; page-break-after: avoid;
}
h3 { font-size: 12pt; margin: 6mm 0 2mm 0; page-break-after: avoid; }
h4 { font-size: 11pt; margin: 4mm 0 1.5mm 0; page-break-after: avoid; }
p { margin: 0 0 2.4mm 0; }
ul, ol { margin: 0 0 2.6mm 0; padding-left: 7mm; }
li { margin-bottom: 1.2mm; }
code {
  font-family: "Cascadia Mono", "Consolas", monospace; font-size: 9.5pt;
  background: #f2f4f7; padding: 0.3mm 1.1mm; border-radius: 1.5pt;
  color: #b03060;
}
pre {
  background: #f7f8fa; border: 0.6pt solid #dde2e8; border-radius: 2pt;
  padding: 3mm 3.5mm; margin: 2.5mm 0 3mm 0; overflow-x: auto;
  line-height: 1.45; page-break-inside: avoid;
}
pre code { background: none; padding: 0; color: #1a1a1a; font-size: 8.8pt; }
table {
  border-collapse: collapse; width: 100%; margin: 2.5mm 0 4mm 0;
  font-size: 9.6pt;
  /* 短表整块不拆；超过 10 行的表才允许跨页（见 _table 的 class）。
     一律 avoid 会在大表放不下时留大片空白，一律 auto 又会把短表劈成两页。 */
  page-break-inside: avoid;
}
table.long { page-break-inside: auto; }
thead { display: table-header-group; }
tr { page-break-inside: avoid; }
th, td { border: 0.6pt solid #c9d0d8; padding: 1.35mm 2.2mm; vertical-align: top; }
th { background: #eef2f7; font-family: "SimHei", sans-serif; font-weight: normal; }
tr:nth-child(even) td { background: #fafbfc; }
blockquote {
  margin: 3mm 0; padding: 2mm 3mm; background: #fdf8e8;
  border-left: 3pt solid #d9b64a; font-size: 10.2pt;
}
blockquote p { margin: 0 0 1mm 0; }
hr { border: none; border-top: 0.6pt dashed #c0c7d0; margin: 5mm 0; }
a { color: #1a4d8f; text-decoration: none; }
img { max-width: 100%; display: block; margin: 3mm auto; }
strong { font-family: "SimHei", sans-serif; font-weight: normal; }
"""


def _inline(s: str) -> str:
    """行内元素。顺序很重要：先转义，再保护反引号代码，最后才是链接/图片。"""
    s = html.escape(s)
    codes: list[str] = []

    def _stash(m: re.Match) -> str:
        codes.append(m.group(1))
        return f"\x00{len(codes) - 1}\x00"

    s = re.sub(r"`([^`]+)`", _stash, s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
    # 图片必须在链接之前匹配，否则 ![a](b) 里的 [a](b) 先被吃掉
    s = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", r'<img alt="\1" src="\2">', s)
    s = re.sub(r"(?<!!)\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', s)
    for idx, c in enumerate(codes):
        s = s.replace(f"\x00{idx}\x00", f"<code>{c}</code>")
    return s


def _table(header: list[str], rows: list[list[str]]) -> str:
    # 行数多才允许跨页；短表整块保留，避免"只剩一行 + 重复表头"这种难看的分页
    cls = " class='long'" if len(rows) > 10 else ""
    out = [f"<table{cls}><thead><tr>"]
    out += [f"<th>{_inline(c)}</th>" for c in header]
    out.append("</tr></thead><tbody>")
    for r in rows:
        out.append("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def md_to_html(md: str) -> str:
    lines = md.replace("\r\n", "\n").split("\n")
    out: list[str] = []
    i, n = 0, len(lines)
    in_code = False
    code_buf: list[str] = []
    para: list[str] = []

    def flush_para() -> None:
        if para:
            out.append("<p>" + _inline(" ".join(para)) + "</p>")
            para.clear()

    while i < n:
        line = lines[i]

        # 代码块
        if line.strip().startswith("```"):
            if not in_code:
                flush_para()
                in_code = True
                code_buf = []
            else:
                in_code = False
                out.append("<pre><code>" + html.escape("\n".join(code_buf)) + "</code></pre>")
            i += 1
            continue
        if in_code:
            code_buf.append(line)
            i += 1
            continue

        # 表格：需要紧随其后的分隔行
        if line.strip().startswith("|") and i + 1 < n and re.match(
            r"^\|[\s:\-\|]+\|$", lines[i + 1].strip()
        ):
            flush_para()
            header = [c.strip() for c in line.strip().strip("|").split("|")]
            i += 2
            rows = []
            while i < n and lines[i].strip().startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            out.append(_table(header, rows))
            continue

        # 标题
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            flush_para()
            lvl = min(len(m.group(1)), 4)
            out.append(f"<h{lvl}>{_inline(m.group(2).strip())}</h{lvl}>")
            i += 1
            continue

        # 分割线
        if re.match(r"^\s*(---|\*\*\*)\s*$", line):
            flush_para()
            out.append("<hr>")
            i += 1
            continue

        # 引用
        if line.strip().startswith(">"):
            flush_para()
            buf = []
            while i < n and lines[i].strip().startswith(">"):
                buf.append(lines[i].strip()[1:].strip())
                i += 1
            out.append("<blockquote><p>" + _inline(" ".join(buf)) + "</p></blockquote>")
            continue

        # 列表
        if re.match(r"^\s*([-*+]|\d+\.)\s+", line):
            flush_para()
            ordered = bool(re.match(r"^\s*\d+\.\s+", line))
            tag = "ol" if ordered else "ul"
            items: list[str] = []
            while i < n:
                m0 = re.match(r"^\s*([-*+]|\d+\.)\s+(.*)$", lines[i])
                if m0:
                    items.append(m0.group(2).strip())
                    i += 1
                elif items and lines[i].strip() and lines[i].startswith("  "):
                    # 续行（缩进的非空行）并入上一项。
                    # 不处理这个的话，中文列表里常见的折行会被切成独立段落，
                    # 结果是每一项被拆成"标题行 + 正文段"，编号还会重复成 1. 1. 1.
                    items[-1] += " " + lines[i].strip()
                    i += 1
                else:
                    break
            out.append(f"<{tag}>" + "".join(f"<li>{_inline(t)}</li>" for t in items) + f"</{tag}>")
            continue

        # 空行 -> 结束段落
        if not line.strip():
            flush_para()
            i += 1
            continue

        para.append(line.strip())
        i += 1

    flush_para()
    return "\n".join(out)


def find_browser() -> str:
    cands = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    ]
    for c in cands:
        if os.path.exists(c):
            return c
    for name in ("msedge", "chrome", "chromium", "chromium-browser", "google-chrome"):
        p = shutil.which(name)
        if p:
            return p
    raise SystemExit("找不到 Edge/Chrome，无法渲染 PDF")


def render(md_path: str, pdf_path: str) -> None:
    md_path = os.path.abspath(md_path)
    pdf_path = os.path.abspath(pdf_path)
    work = os.path.dirname(md_path)

    with open(md_path, encoding="utf-8") as f:
        body = md_to_html(f.read())

    title = os.path.splitext(os.path.basename(md_path))[0]
    doc = (
        "<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>"
        f"<title>{html.escape(title)}</title><style>{CSS}</style></head>"
        f"<body>{body}</body></html>"
    )
    html_path = os.path.join(work, "_render.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(doc)

    browser = find_browser()
    url = "file:///" + html_path.replace("\\", "/").replace(" ", "%20")
    # --no-pdf-header-footer 去掉浏览器自带的日期/URL 页眉页脚
    cmd = [
        browser,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--no-pdf-header-footer",
        f"--print-to-pdf={pdf_path}",
        url,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if not os.path.exists(pdf_path):
        print(r.stdout[-2000:], r.stderr[-2000:], file=sys.stderr)
        raise SystemExit("PDF 未生成")
    print(f"[pdf] {pdf_path} ({os.path.getsize(pdf_path)/1024:.0f} KB)")


def stamp(pdf_path: str, title: str) -> None:
    """补页码 + 元数据 + 书签。PyMuPDF 不在就静默跳过。"""
    try:
        import pymupdf
    except ImportError:
        print("[stamp] 跳过（没有 PyMuPDF）")
        return

    doc = pymupdf.open(pdf_path)
    total = doc.page_count

    # 粗体标题 -> 书签。
    # 注意：中文标题在 PDF 里常被拆成"一个字一个 span"，所以必须按**行**合并，
    # 不能按 span 取——否则一个 14 字的标题会变成 14 个书签（我第一版就是这样）。
    headings: list[tuple[int, str]] = []
    for pno in range(total):
        page = doc[pno]
        for blk in page.get_text("dict")["blocks"]:
            for line in blk.get("lines", []):
                big = [s for s in line.get("spans", []) if s["size"] >= 13.5]
                if not big:
                    continue
                text = "".join(s["text"] for s in big).strip()
                if text and len(text) <= 60:
                    headings.append((pno + 1, text))

    for pno in range(total):
        page = doc[pno]
        text = f"第 {pno + 1} 页 / 共 {total} 页"
        w = page.rect.width
        page.insert_text(
            (w / 2 - len(text) * 2.2, page.rect.height - 24),
            text,
            fontsize=8,
            fontname="china-s",
            color=(0.45, 0.45, 0.45),
        )

    # 首页不加页码（保持干净）
    doc.set_metadata({"title": title, "author": "sunever", "creator": "md2pdf.py"})

    seen = set()
    toc = []
    for pno, text in headings:
        key = (text, pno)
        if key in seen or len(text) > 60:
            continue
        seen.add(key)
        toc.append([1, text, pno])
    if toc:
        doc.set_toc(toc)

    doc.saveIncr()
    doc.close()
    print(f"[stamp] {total} 页，{len(toc)} 个书签")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("md")
    ap.add_argument("pdf")
    ap.add_argument("--title", default=None)
    ap.add_argument("--no-stamp", action="store_true")
    a = ap.parse_args()

    render(a.md, a.pdf)
    if not a.no_stamp:
        title = a.title or os.path.splitext(os.path.basename(a.md))[0]
        stamp(os.path.abspath(a.pdf), title)


if __name__ == "__main__":
    main()