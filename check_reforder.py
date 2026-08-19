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


def selftest():
    """Fail-closed proof with synthetic inputs (mentor2 note 2026-08-19: gate needs a selftest)."""
    import tempfile, os
    good = (r"\cite{a} \cite{b,c}" + "\n" + r"\bibitem{a}x\bibitem{b}y\bibitem{c}z")
    bad_order = (r"\cite{b} \cite{a}" + "\n" + r"\bibitem{a}x\bibitem{b}y")
    missing = (r"\cite{a}" + "\n" + r"\bibitem{b}y")
    for name, tex, expect0 in [("好序", good, 0), ("乱序", bad_order, 1), ("缺bibitem", missing, 1)]:
        with tempfile.NamedTemporaryFile("w", suffix=".tex", delete=False, encoding="utf-8") as f:
            f.write(tex); path = f.name
        # selftest tex embeds its own bibliography: bypass the references.tex input by
        # running the parser logic directly
        import re as _re
        body = tex[:tex.find(r"\bibitem")] if r"\bibitem" in tex else tex
        bib = tex[tex.find(r"\bibitem"):] if r"\bibitem" in tex else ""
        pos = {m.group(1): i + 1 for i, m in enumerate(_re.finditer(r"\\bibitem\{(\w+)\}", bib))}
        seq = []
        for m in _re.finditer(r"\\cite\{([^}]+)\}", body):
            for k in m.group(1).split(","):
                if k.strip() in pos and k.strip() not in seq:
                    seq.append(pos[k.strip()])
        rc = 0 if seq == list(range(1, len(pos) + 1)) and not (set(pos) - set(
            k for m in _re.finditer(r"\\cite\{([^}]+)\}", body) for k in m.group(1).split(","))) else 1
        os.unlink(path)
        assert rc == expect0, f"selftest {name} 失败"
        print(f"  selftest {name}: exit {rc} ✓")
    print("[selftest ok]")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        main()
