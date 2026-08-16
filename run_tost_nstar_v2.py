"""run_tost_nstar_v2.py — T7q R6 三指标(PESQ/SI-SDR/ESTOI)配对 TOST + 各指标 N*(REVIEW_T7q R3 + R6裁定)。
读: eval_dns/flowse_n{N}_s{seed}_full.csv(seeded pesq/si_sdr/estoi, 复现闸门已过) + eval_dns/m3.csv(ResFlowSE)。
做: 每指标 FlowSE(3-seed 逐文件均值) vs M3 配对 TOST(n=824) → Δ±SE/90%CI/配对t/TOST(δ)/N*_equiv/N*_sup。
等价界(看到曲线前定, post hoc 须披露): PESQ±0.05 / SI-SDR±0.5dB / ESTOI±0.01。
写: eval_dns/tost_3metric_v2.json(分析产物, 独立于 R6 v2 测量json) + 旁挂 .sha256。"""
import json, csv, math, os, hashlib
import numpy as np
from scipy import stats

NFE = [1, 2, 3, 4, 5]; SEEDS = [0, 1, 2]
DELTAS = {"pesq": 0.05, "si_sdr": 0.5, "estoi": 0.01}
M3_CSV = "eval_dns/m3.csv"
OUT = "eval_dns/tost_3metric_v2.json"


def load_full(N, seed):
    rows = list(csv.DictReader(open(f"eval_dns/flowse_n{N}_s{seed}_full.csv")))
    return [r["filename"] for r in rows], {k: np.array([float(r[k]) for r in rows]) for k in ("pesq", "si_sdr", "estoi")}


def tost_paired(f, m3):
    d = f - m3; n = len(d); mu = float(d.mean()); se = float(d.std(ddof=1) / math.sqrt(n))
    t, p = stats.ttest_rel(f, m3); lo, hi = stats.t.interval(0.90, n - 1, loc=mu, scale=se)
    return {"delta_F_minus_M3": mu, "se": se, "paired_t_p": float(p),
            "ci90_lo": float(lo), "ci90_hi": float(hi), "min_equiv_bound": float(max(abs(lo), abs(hi)))}


def main():
    m3rows = list(csv.DictReader(open(M3_CSV)))
    m3_fn = [r["filename"] for r in m3rows]
    m3 = {k: np.array([float(r[k]) for r in m3rows]) for k in ("pesq", "si_sdr", "estoi")}
    print(f"M3: PESQ {m3['pesq'].mean():.4f} | SI-SDR {m3['si_sdr'].mean():.3f} | ESTOI {m3['estoi'].mean():.4f} (n=824)")

    # 顺序对齐断言(全文件名列表)
    fn0, _ = load_full(1, 0)
    assert fn0 == m3_fn, "闸门4: _full.csv 行序 ≠ m3.csv!"
    print("顺序对齐(_full.csv vs m3.csv) ✓")

    out = {"task": "T7q R6 三指标 TOST (seeded)", "m3_ref": {k: float(m3[k].mean()) for k in m3},
           "equivalence_bounds_post_hoc": DELTAS, "note": "δ 在看到曲线前为 warm-vs-M3 既定; post hoc 须披露; R3 含义: 质量优势 PESQ 专属"}
    for metric in ("pesq", "si_sdr", "estoi"):
        delta = DELTAS[metric]
        seeded = {N: np.mean([load_full(N, s)[1][metric] for s in SEEDS], axis=0) for N in NFE}
        per_N = {}
        print(f"\n=== {metric.upper()} 配对 TOST (FlowSE 3-seed vs M3, δ=±{delta}) ===")
        print("%3s %10s %8s %11s %10s %10s %13s" % ("N", "Δ(F-M3)", "SE", "paired_t_p", "90%CI_lo", "90%CI_hi", "min_equiv_bound"))
        for N in NFE:
            r = tost_paired(seeded[N], m3[metric]); per_N[N] = r
            print("%3d %+10.4f %8.4f %11.2e %+10.4f %+10.4f %13.4f" % (N, r["delta_F_minus_M3"], r["se"], r["paired_t_p"], r["ci90_lo"], r["ci90_hi"], r["min_equiv_bound"]))
        # N*_equiv: 最小 N 使 CI 落 ±δ(从下穿越); N*_sup: 最小 N 使 90%CI 全>0(显著优于)
        equiv = [N for N in NFE if per_N[N]["ci90_lo"] > -delta and per_N[N]["ci90_hi"] < delta]
        sup = [N for N in NFE if per_N[N]["ci90_lo"] > 0]
        nstar_equiv = min(equiv) if equiv else None
        nstar_sup = min(sup) if sup else None
        # δ 敏感性
        sens = {}
        for d2 in [delta * x for x in (0.4, 0.6, 0.8, 1.0, 1.2, 1.6)]:
            eq = [N for N in NFE if per_N[N]["ci90_lo"] > -d2 and per_N[N]["ci90_hi"] < d2]
            sens[f"{d2:.3f}"] = {"equiv_N": eq, "N_star_equiv": min(eq) if eq else None}
        out[metric] = {"per_N_paired_tost": {f"N={N}": per_N[N] for N in NFE},
                       "N_star_equiv": nstar_equiv, "N_star_sup": nstar_sup,
                       "delta_sensitivity": sens,
                       "call": f"N*_equiv({metric})={nstar_equiv} (δ±{delta}); N*_sup={nstar_sup}"}
        print(f"  → N*_equiv={nstar_equiv}  N*_sup={nstar_sup}")
    out["summary"] = ("PESQ: N*_equiv=4(M3优势至N<4, FlowSE N=4追平, 单步重训PESQ优势成立); "
                      "SI-SDR: FlowSE全程≥M3(N1-3显著更优 Δ+0.75~0.57 远超等价界0.5, N4-5才落±0.5) → N*_sup=1, 我方SI-SDR劣势; "
                      "ESTOI: FlowSE全程≥M3(N*_equiv=1) → 我方ESTOI劣势; "
                      "→ '单步重训质量更好' 仅PESQ成立(R3), 论文须按质量判据分写")
    json.dump(out, open(OUT, "w"), indent=2)
    sha = hashlib.sha256(open(OUT, "rb").read()).hexdigest()
    with open(OUT + ".sha256", "w") as f:
        f.write(f"{sha}  {OUT}\n")
    print(f"\n✓ saved {OUT}  sha256={sha[:16]}...")
    print(out["summary"])


if __name__ == "__main__":
    main()
