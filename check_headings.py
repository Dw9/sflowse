#!/usr/bin/env python3
"""Heading-rendering gate (2026-08-19: subsection double-numbering bug, present since v1).

The md carries explicit numbers ("### 3.1 Title"); build_tex strips them so LaTeX
auto-numbers.  If the strip regex misses a form, the built tex gets
\\subsection{3.1 Title} and the PDF prints "3.1. 3.1 Title".  This gate asserts
no sectioning-command title in the built tex starts with a digit.

Usage: python check_headings.py [manuscript.tex]   Exit 1 on violation.
"""
import re
import sys

TEX = sys.argv[1] if len(sys.argv) > 1 else "MDPI-PAPER/manuscript.tex"


def main():
    tex = open(TEX, encoding="utf-8").read()
    bad = [m.group(0)[:60] for m in re.finditer(
        r"\\(?:sub)*section\{(\d|[^}]*?\s\d)[^}]*\}", tex)]
    bad = [b for b in bad if re.match(r"\\(?:sub)*section\{[\d]", b)]
    if bad:
        print("[FAIL] 标题内嵌编号(将双编号渲染):")
        for b in bad:
            print("  ", b)
        sys.exit(1)
    n = len(re.findall(r"\\(?:sub)*section\{", tex))
    print(f"[ok] {n} 个节标题均无内嵌编号")


if __name__ == "__main__":
    main()
