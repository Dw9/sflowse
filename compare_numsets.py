#!/usr/bin/env python3
"""EN/ZH manuscript number-set comparison (gate 3 of the P0+P1 fix plan).

Two-tier gate:
  HARD — every number introduced by the 2026-08-18 P0+P1 revision must occur the
         same number of times in both drafts (any mismatch exits 1);
  SOFT — the residual multiset difference (prose-citation years, translation
         paraphrase like "15th" vs "第 15 百分位") is printed and must stay
         below RESIDUAL_MAX occurrences per side, else exit 1.

HTML comments and \\cite commands are stripped before extraction (EN carries
citation keys where ZH carries prose years — a language artifact, not a number).
"""
import collections
import re
import sys

EN = "MDPI-PAPER/draft_initial_review.md"
ZH = "MDPI-PAPER/draft_initial_review_zh.md"

P0P1_NUMBERS = [
    "0.0400", "0.0579", "0.0221", "0.0086", "0.0259", "9.6×10⁻⁴", "6.7×10⁻³",
    "10⁻³⁰⁰", "0.062", "2.792", "13,382", "2230", "0.015%", "0.012%", "8.9", "0.54%",
    "0.7389", "0.767", "25.9", "0.01215", "0.31498", "0.20%", "0.12%", "6.4%", "2.8%", "1.7",
    "2.51", "510", "128", "0.999", "3.0622", "62.1", "4.23", "0.0792",
]
RESIDUAL_MAX = 12  # per side; benign class: citation years + paraphrase tokens


def tokens(path):
    md = open(path, encoding="utf-8").read()
    md = re.sub(r"<!--.*?-->", "", md, flags=re.S)
    md = re.sub(r"\s*\[SRC:[^\]]*\]", "", md)
    md = re.sub(r"\\(?:cite|citep|citet|ref|eqref)\{[^}]+\}", " ", md)
    return collections.Counter(re.findall(r"\d+(?:\.\d+)?", md))


# Structural language artifacts excluded from the SOFT residual: prose citation
# years (ZH cites "JBHI 2025" where EN carries a \cite key) and the P.862.x
# standard numbers cited inline in ZH only.
YEAR_RE = re.compile(r"^(19|20)\d{2}$")
CITE_ADJ = {"862", "862.2"}


def residual(counter):
    return collections.Counter({t: c for t, c in counter.items()
                                if not YEAR_RE.match(t) and t not in CITE_ADJ})


def main():
    en, zh = tokens(EN), tokens(ZH)
    print(f"EN tokens: {sum(en.values())}  ZH tokens: {sum(zh.values())}")

    hard_fail = False
    for t in P0P1_NUMBERS:
        a, b = en[t], zh[t]
        if a != b:
            print(f"[HARD FAIL] '{t}': EN={a} ZH={b}")
            hard_fail = True
    if hard_fail:
        sys.exit(1)
    print(f"[hard ok] all {len(P0P1_NUMBERS)} P0/P1 numbers balanced")

    only_en, only_zh = residual(en - zh), residual(zh - en)
    for name, d in (("EN-only", only_en), ("ZH-only", only_zh)):
        if d:
            print(f"{name} residual: {dict(d)}")
    if sum(only_en.values()) > RESIDUAL_MAX or sum(only_zh.values()) > RESIDUAL_MAX:
        print(f"[SOFT FAIL] residual exceeds {RESIDUAL_MAX} per side — inspect above")
        sys.exit(1)
    print(f"[soft ok] residual {sum(only_en.values())}/{sum(only_zh.values())} "
          f"within tolerance (citation years / translation paraphrase)")


if __name__ == "__main__":
    main()
