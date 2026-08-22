#!/usr/bin/env python3
"""fail-closed 表/图编号门禁(kimi 模板审 FAIL 修复配套,铁律14):
  稿件(manuscript.tex, 构建产物)中 ——
  A. 物理序: \\caption 在 table/figure 环境中的出现序 = 1..N 严格(LaTeX 自动编号即此);
  B. 首引序: 正文+表注+单元格(含复数 "Tables 7 and 9"/"Tables 7, 9")中每个编号的首次出现
     必须按 1,2,3,... 递增 —— 任何前引(先提大号后提小号)必红。
  模板依据: MDPI template L211/L222(图表置于首次引用处)+ L477-480(编号按出现序,含表内引用)。
用法: python3 check_taborder.py [tex路径] [--selftest]
"""
import re, sys, tempfile, subprocess, os

TEX = "MDPI-PAPER/manuscript.tex"


def parse(tex):
    lines = tex.splitlines()
    caps, mentions = [], []  # (kind, number) 按出现序
    env = None
    for ln in lines:
        m = re.match(r"\s*\\begin\{(table|figure)\}", ln)
        if m:
            env = m.group(1)
        if re.search(r"\\" + "caption\{", ln) and env:
            caps.append(env)
        if re.match(r"\s*\\end\{(table|figure)\}", ln):
            env = None
        # 提及(含表注/单元格文本; 逐行扫, 忽略 latex 命令头)
        for m2 in re.finditer(r"\b(Tables|Figures|Table|Figure|Figs\.|Fig\.)\s*\.?\s*([0-9]+(?:\s*[,–-]\s*[0-9]+)*(?:\s+and\s+[0-9]+)?)", ln):
            kind = "table" if m2.group(1).lower().startswith("table") else "figure"
            body = m2.group(2)
            for num in re.findall(r"[0-9]+", body):
                mentions.append((kind, int(num), ln.strip()[:60]))
    return caps, mentions


def first_mention_seq(mentions, kind):
    seen, seq = set(), []
    for k, n, ctx in mentions:
        if k != kind or n in seen:
            continue
        seen.add(n)
        seq.append((n, ctx))
    return seq


def run(tex_path):
    tex = open(tex_path, encoding="utf-8").read()
    caps, mentions = parse(tex)
    bad = []
    for kind in ("table", "figure"):
        seq = first_mention_seq(mentions, kind)
        nums = [n for n, _ in seq]
        expect = list(range(1, len(nums) + 1))
        if nums != expect:
            for i, (n, ctx) in enumerate(seq):
                if n != i + 1:
                    bad.append(f"{kind} first-mention #{i+1} is {n}, expected {i+1} | ctx: {ctx}")
                    break
        n_phys = caps.count(kind)
        if len(nums) != n_phys:
            bad.append(f"{kind}: {n_phys} captions vs {len(nums)} distinct numbers mentioned "
                       f"(uncited table/figure or stale number)")
    if bad:
        print("[FAIL] taborder gate:")
        for b in bad:
            print("   ", b)
        return 1
    print(f"[ok] tables 1..{caps.count('table')} and figures 1..{caps.count('figure')}: "
          f"first-mention order strict, all cited")
    return 0


def selftest():
    ok = True
    with tempfile.NamedTemporaryFile("w", suffix=".tex", delete=False) as f:
        f.write("\\begin{table}\\caption{a}\\end{table}\nsee Table 2 here\n\\begin{table}\\caption{b}\\end{table}\nthen Table 1\n")
        t1 = f.name
    r = subprocess.run([sys.executable, __file__, t1], capture_output=True, text=True)
    if r.returncode == 0:
        print("[SELFTEST FAIL] 前引(2先于1)应非零退出"); ok = False
    else:
        print(f"[SELFTEST PASS] 前引被拦 exit={r.returncode}")
    with tempfile.NamedTemporaryFile("w", suffix=".tex", delete=False) as f:
        f.write("\\begin{table}\\caption{a}\\end{table}\nsee Table 1 here\n\\begin{table}\\caption{b}\\end{table}\nthen Table 2 — but a third table exists uncited\n\\begin{table}\\caption{c}\\end{table}\n")
        t2 = f.name
    r2 = subprocess.run([sys.executable, __file__, t2], capture_output=True, text=True)
    if r2.returncode == 0:
        print("[SELFTEST FAIL] 有表未被引用应非零退出"); ok = False
    else:
        print(f"[SELFTEST PASS] 未引用表被拦 exit={r2.returncode}")
    with tempfile.NamedTemporaryFile("w", suffix=".tex", delete=False) as f:
        f.write("see Fig. 2 here\n\\begin{figure}\\caption{a}\\end{figure}\nFigure 1 later\n\\begin{figure}\\caption{b}\\end{figure}\n")
        t3 = f.name
    r3 = subprocess.run([sys.executable, __file__, t3], capture_output=True, text=True)
    if r3.returncode == 0:
        print("[SELFTEST FAIL] 缩写前引(Fig. 2 先于 Figure 1)应非零退出"); ok = False
    else:
        print(f"[SELFTEST PASS] 缩写前引被拦 exit={r3.returncode}")
    os.unlink(t1); os.unlink(t2); os.unlink(t3)
    return 0 if ok else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    path = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else TEX
    sys.exit(run(path))
