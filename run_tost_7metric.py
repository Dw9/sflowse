"""run_tost_7metric.py — T7q R6 七指标全表(REVIEW_R6 R6.1/R6.2): pesq/si_sdr/estoi(有预注册等价界→TOST) + dnsmos_sig/bak/ovrl/p808(无等价界→只报显著性)。
读: eval_dns/flowse_n{N}_s{seed}_full.csv + eval_dns/m3.csv(均 8 列含 dnsmos_*/p808)。seed 平均逐文件配对 n=824。
写: eval_dns/tost_7metric_v2.json + 旁挂 .sha256(仓库根相对路径, R6.4)。"""
import json, csv, math, hashlib
import numpy as np
from scipy import stats

NFE = [1, 2, 3, 4, 5]; SEEDS = [0, 1, 2]
EQ_BOUNDS = {"pesq": 0.05, "si_sdr": 0.5, "estoi": 0.01}   # 有预注册等价界
SIG_ONLY = ["dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl", "p808"]  # 无等价界, 只报显著性
ALLM = ["pesq", "si_sdr", "estoi"] + SIG_ONLY
OUT = "eval_dns/tost_7metric_v2.json"


def load_full(N, seed):
    rows = list(csv.DictReader(open(f"eval_dns/flowse_n{N}_s{seed}_full.csv")))
    return [r["filename"] for r in rows], {k: np.array([float(r[k]) for r in rows]) for k in ALLM}


def paired(f, m3):
    d = f - m3; n = len(d); mu = float(d.mean()); se = float(d.std(ddof=1) / math.sqrt(n))
    t, p = stats.ttest_rel(f, m3); lo, hi = stats.t.interval(0.90, n - 1, loc=mu, scale=se)
    return {"delta_F_minus_M3": mu, "rel_pct": float(mu / m3.mean() * 100) if m3.mean() else None,
            "se": se, "paired_t_p": float(p), "ci90_lo": float(lo), "ci90_hi": float(hi)}


def main():
    m3rows = list(csv.DictReader(open("eval_dns/m3.csv")))
    m3_fn = [r["filename"] for r in m3rows]
    m3 = {k: np.array([float(r[k]) for r in m3rows]) for k in ALLM}
    fn0, _ = load_full(1, 0)
    assert fn0 == m3_fn, "行序 ≠ m3.csv"
    print("M3:", {k: round(float(m3[k].mean()), 4) for k in ALLM}, "(n=824); 顺序对齐 ✓")

    seeded = {N: {k: np.mean([load_full(N, s)[1][k] for s in SEEDS], axis=0) for k in ALLM} for N in NFE}
    out = {"task": "T7q R6 七指标配对(FlowSE 3-seed vs M3, n=824)", "m3_ref": {k: float(m3[k].mean()) for k in ALLM},
           "equivalence_bounds_post_hoc": EQ_BOUNDS, "sig_only_no_equiv_bound": SIG_ONLY}
    for metric in ALLM:
        per_N = {}
        for N in NFE:
            r = paired(seeded[N][metric], m3[metric]); per_N[N] = r
        if metric in EQ_BOUNDS:
            delta = EQ_BOUNDS[metric]
            equiv = [N for N in NFE if per_N[N]["ci90_lo"] > -delta and per_N[N]["ci90_hi"] < delta]
            sup = [N for N in NFE if per_N[N]["ci90_lo"] > 0]
            out[metric] = {"per_N": {f"N={N}": per_N[N] for N in NFE}, "equiv_bound": delta,
                           "N_star_equiv": min(equiv) if equiv else None, "N_star_sup": min(sup) if sup else None}
            print(f"\n=== {metric} (δ=±{delta}) N*_equiv={out[metric]['N_star_equiv']} N*_sup={out[metric]['N_star_sup']} ===")
        else:
            for N in NFE:
                p = per_N[N]["paired_t_p"]
                per_N[N]["verdict"] = "FlowSE_signif_better" if per_N[N]["ci90_lo"] > 0 else ("M3_signif_better" if per_N[N]["ci90_hi"] < 0 else "n.s.(α0.05)")
            out[metric] = {"per_N": {f"N={N}": per_N[N] for N in NFE}, "equiv_bound": None,
                           "note": "无预注册等价界 → 只报显著性, 不作等价/非等价判定"}
            print(f"\n=== {metric} (无等价界, 只报显著性) ===")
        for N in NFE:
            r = per_N[N]
            extra = f" {r.get('verdict','')}" if metric in SIG_ONLY else ""
            print("  N=%d Δ=%+.4f (%.2f%%) p=%.2e 90%%CI[%+.4f,%+.4f]%s" % (N, r["delta_F_minus_M3"], r["rel_pct"], r["paired_t_p"], r["ci90_lo"], r["ci90_hi"], extra))

    out["R3_refined"] = ("七指标@N=1: 我方(M3)优 = PESQ(−0.173,5.66%大) + DNSMOS-SIG(−0.009,0.26%统计显著实际可忽略); "
                         "FlowSE优 = SI-SDR(+0.754超界非等价)/DNSMOS-BAK/DNSMOS-OVRL/P808; "
                         "平 = ESTOI(±0.01界内等价). → 优势落在语音信号失真类(PESQ大+SIG可忽略); 输波形保真/背景抑制/整体MOS; 可懂度平. "
                         "非单指标孤例(PESQ+SIG同类)→堵'单指标伪影'攻击; 但SIG幅度0.26%不得抬成第二支柱.")
    json.dump(out, open(OUT, "w"), indent=2)
    sha = hashlib.sha256(open(OUT, "rb").read()).hexdigest()
    with open(OUT + ".sha256", "w") as f:
        f.write(f"{sha}  {OUT}\n")   # 仓库根相对路径(R6.4)
    print(f"\n✓ saved {OUT}  sha256={sha[:16]}...")


if __name__ == "__main__":
    main()
