#!/usr/bin/env python3
"""Reference first-appearance order gate (mentor2 R1-F4, 2026-08-19).

MDPI template line 477: references must be numbered in order of appearance in
the text (including citations in tables and legends).  This gate reads the
BUILT .tex, maps every \\cite key to its \\bibitem position, extracts the
sequence of first appearances, and asserts it equals 1..N strictly.

Usage: python check_reforder.py [manuscript.tex]   (default MDPI-PAPER/manuscript.tex)
Exit 1 on violation (fail-closed).
"""
import re
import sys

TEX = sys.argv[1] if len(sys.argv) > 1 else "MDPI-PAPER/manuscript.tex"


def main():
    tex = open(TEX, encoding="utf-8").read()
    body = tex  # built tex carries \cite commands; bibliography arrives via \input{references}

    refs = open("MDPI-PAPER/references.tex", encoding="utf-8").read()
    pos = {m.group(1): i + 1 for i, m in enumerate(re.finditer(r"\\bibitem\{(\w+)\}", refs))}

    first_seen, seq = set(), []
    for m in re.finditer(r"\\cite\{([^}]+)\}", body):
        for k in (s.strip() for s in m.group(1).split(",")):
            if k not in pos:
                print(f"[FAIL] cite key with no bibitem: {k}")
                sys.exit(1)
            if k not in first_seen:
                first_seen.add(k)
                seq.append(pos[k])

    n = len(pos)
    missing = set(pos) - first_seen
    if missing:
        print(f"[FAIL] bibitems never cited: {sorted(missing)}")
        sys.exit(1)
    if seq != list(range(1, n + 1)):
        bad = [(i + 1, v) for i, v in enumerate(seq) if v != i + 1][:5]
        print(f"[FAIL] first-appearance sequence not monotonic: {seq[:10]}...; first mismatches {bad}")
        sys.exit(1)
    print(f"[ok] reference first-appearance order = 1..{n} strictly (template L477 satisfied)")


if __name__ == "__main__":
    main()
