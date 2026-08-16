#!/usr/bin/env python3
"""配对显著性检验：读 eval_dns/*.csv 逐文件分数，按 filename 对齐，
对关键对比做配对 t 检验 + Wilcoxon 符号秩检验（非参数备份）。
铁律5/10：只用全 824 逐文件真值，结果供论文引用（需入 ledger 标源）。
用法: python run_sigtest.py
"""
import os, json
import numpy as np
import pandas as pd
from scipy import stats

CSV = "eval_dns"
METRICS = ["pesq", "si_sdr", "estoi"]
# (A, B): 检验 A 相对 B 的差异
COMPARISONS = [
    ("warm", "cold",      "warm-start 是否显著优于随机初始化（主张支柱）"),
    ("m3", "flowse_n5",   "M3 是否与 5 步 FlowSE 接近（差异不显著=接近）"),
    ("m3", "flowse_n1",   "M3 是否显著优于 naive N=1 截断"),
    ("m3", "m1",          "EA 增益是否显著（M3 vs 无EA，反驳 R2 'EA incremental'）"),
    ("m3", "m2",          "残差增益是否显著（M3 vs 无residual）"),
]

def load(name):
    p = os.path.join(CSV, f"{name}.csv")
    if not os.path.exists(p):
        return None
    return pd.read_csv(p).set_index("filename")

def main():
    need = sorted({n for c in COMPARISONS for n in c[:2]})
    dfs = {n: load(n) for n in need}
    ready = [n for n, d in dfs.items() if d is not None]
    pending = [n for n, d in dfs.items() if d is None]
    print(f"[CSV] ready={ready}  pending={pending}")
    out = {"note": "paired tests on full-824 per-file scores (eval_dns/*.csv)", "comparisons": []}
    for A, B, desc in COMPARISONS:
        if dfs.get(A) is None or dfs.get(B) is None:
            print(f"\n=== {A} vs {B} — SKIP (CSV 未就绪) ===")
            continue
        dA, dB = dfs[A], dfs[B]
        common = dA.index.intersection(dB.index)
        print(f"\n=== {A} vs {B} ({len(common)} 对齐文件) — {desc} ===")
        rec = {"A": A, "B": B, "desc": desc, "n": int(len(common)), "metrics": {}}
        for m in METRICS:
            a, b = dA.loc[common, m].values, dB.loc[common, m].values
            diff = a - b
            t, pt = stats.ttest_rel(a, b)
            try:
                w, pw = stats.wilcoxon(a, b)
            except ValueError:
                w, pw = float("nan"), float("nan")
            sig = "***" if pt < 0.001 else "**" if pt < 0.01 else "*" if pt < 0.05 else "n.s."
            print(f"  {m:>7s}: Δ={diff.mean():+.4f} (A={a.mean():.4f} B={b.mean():.4f}) "
                  f"| t={t:+.3f} p_t={pt:.2e} {sig} | Wilcoxon p={pw:.2e}")
            rec["metrics"][m] = {"mean_A": float(a.mean()), "mean_B": float(b.mean()),
                                 "mean_diff": float(diff.mean()), "t": float(t),
                                 "p_ttest": float(pt), "p_wilcoxon": float(pw), "sig": sig}
        out["comparisons"].append(rec)
    with open("sigtest_results.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\n>>> 已保存 sigtest_results.json（供 ledger 标源引用）")

if __name__ == "__main__":
    main()
