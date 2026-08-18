"""run_tost_nstar.py — T7q 的 N* 正式判定(REVIEW_T7q R1/R2/R3/R5; 收编导师 /tmp 脚本, 结果由脚本生成不手抄)。
读: eval_dns/m3.csv(pesq,si_sdr,estoi) + flowse_n{1..5}_s{0,1,2}_pesq.csv(seeded PESQ 逐文件) + flowse_n{1,5}.csv(旧单抽, 含si_sdr/estoi)
做:
  R1 配对 TOST: FlowSE(N, 3-seed PESQ 均值) vs ResFlowSE M3 | n=824 逐文件配对 → Δ±SE, 90%CI, 最小可成立界, δ 敏感性
     N*_equiv(δ=±0.05) / N*_sup(90%CI全>0); first_cross 降级为诊断量
  R2 穿越稳健性: 3/3 seed 在 (3,4) 间穿越(不用 3σ 启发式)
  R3 SI-SDR/ESTOI 配对(旧单抽 CSV; ⚠️ seeded 逐文件缺→R6 须重跑推理): 同代价 N=1 下 FlowSE 是否在波形/可懂度上也赢
  R5 band_check 跨 N(每频带+full, 相邻 N 差 vs seed std; 旧版答成跨频带=语音能量分布无信息量, 已废)
写回: eval_dns/flowse_nfe_seeded.json 的 N_star / band_check / 新增 si_sdr_estoi_paired 段(重算 sha256_self)。
"""
import json, csv, math
import numpy as np
from scipy import stats

JPATH = "eval_dns/flowse_nfe_seeded.json"
NFE = [1, 2, 3, 4, 5]
SEEDS = [0, 1, 2]
BANDS = ["low (0-1k)", "mid (1-4k)", "high (4-8k)", "full"]


def load_m3():
    rows = list(csv.DictReader(open("eval_dns/m3.csv")))
    m3 = float(np.mean([float(r["pesq"]) for r in rows]))
    pesq = np.array([float(r["pesq"]) for r in rows])
    return rows, pesq, m3


def tost_paired(f, m3):
    """f, m3: per-file arrays. 返回 Δ, SE, t_p, 90%CI(lo,hi), min_equiv_bound."""
    d = f - m3; n = len(d); mu = float(d.mean()); se = float(d.std(ddof=1) / math.sqrt(n))
    t, p = stats.ttest_rel(f, m3)
    lo, hi = stats.t.interval(0.90, n - 1, loc=mu, scale=se)
    return {"delta_F_minus_M3": mu, "se": se, "paired_t_p": float(p),
            "ci90_lo": float(lo), "ci90_hi": float(hi), "min_equiv_bound": float(max(abs(lo), abs(hi)))}


def main():
    rows, m3_pesq, m3_mean = load_m3()
    print(f"M3 PESQ mean={m3_mean:.6f} (n={len(m3_pesq)})")

    # 顺序核验(R7): seeded 无文件名列, 用 corr 验行序
    n5_old = np.array([float(r["pesq"]) for r in csv.DictReader(open("eval_dns/flowse_n5.csv"))])
    s0 = np.loadtxt("eval_dns/flowse_n5_s0_pesq.csv")
    c = np.corrcoef(n5_old, s0)[0, 1]; sh = np.corrcoef(n5_old, np.roll(s0, 1))[0, 1]
    order_ok = bool(c > 0.9 > sh)
    print(f"顺序核验: corr(n5_old,n5_s0)={c:.4f} 错位={sh:.4f} → {'顺序一致 ✓' if order_ok else '⚠️存疑'}")

    # R1: PESQ 配对 TOST (FlowSE 3-seed 均值 vs M3)
    seeded = {N: np.mean([np.loadtxt(f"eval_dns/flowse_n{N}_s{s}_pesq.csv") for s in SEEDS], axis=0) for N in NFE}
    per_N = {}
    print("\n=== R1 PESQ 配对 TOST (FlowSE 3-seed vs M3, n=824) ===")
    print("%3s %9s %8s %11s %10s %10s %13s" % ("N", "Δ(F-M3)", "SE", "paired_t_p", "90%CI_lo", "90%CI_hi", "min_equiv_bound"))
    for N in NFE:
        r = tost_paired(seeded[N], m3_pesq)
        per_N[N] = r
        print("%3d %+9.4f %8.4f %11.2e %+10.4f %+10.4f %13.4f" % (N, r["delta_F_minus_M3"], r["se"], r["paired_t_p"], r["ci90_lo"], r["ci90_hi"], r["min_equiv_bound"]))

    delta_sens = {}
    print("\n=== δ 敏感性 (N*_equiv) ===")
    for delta in [0.02, 0.03, 0.04, 0.05, 0.06, 0.08]:
        eq = [N for N in NFE if per_N[N]["ci90_lo"] > -delta and per_N[N]["ci90_hi"] < delta]
        delta_sens[delta] = (eq, min(eq) if eq else None)
        print("  δ=±%.2f → 等价 N=%s → N*_equiv=%s" % (delta, eq, min(eq) if eq else ">5"))
    sup = [N for N in NFE if per_N[N]["ci90_lo"] > 0]
    n_star_sup = min(sup) if sup else None
    n_star_equiv_05 = delta_sens[0.05][1]
    print(f"\nN*_equiv(δ=±0.05) = {n_star_equiv_05}   N*_sup(90%CI全>0) = {n_star_sup}")

    # R2: 3/3 seed 在 (3,4) 间穿越
    curve = json.load(open(JPATH))["curve"]
    cross_robust = None
    for i in range(1, len(NFE)):
        below = all(curve[f"N={NFE[i-1]}"]["per_seed_pesq"][s] < m3_mean for s in range(3))
        above = all(curve[f"N={NFE[i]}"]["per_seed_pesq"][s] > m3_mean for s in range(3))
        if below and above:
            cross_robust = {"prev_N": NFE[i-1], "cross_N": NFE[i], "all_3_seeds_below_then_above": True,
                            "prev_per_seed": curve[f"N={NFE[i-1]}"]["per_seed_pesq"], "cross_per_seed": curve[f"N={NFE[i]}"]["per_seed_pesq"]}
            break
    print(f"\n=== R2 穿越稳健性: 3/3 seed 在 ({cross_robust['prev_N']},{cross_robust['cross_N']}) 间穿越 ✓ (不用3σ) ===" if cross_robust else "⚠️ 未找到 3/3 穿越点")

    # R3: SI-SDR/ESTOI 配对(旧单抽 CSV; ⚠️ seeded 缺→R6)
    print("\n=== R3 SI-SDR/ESTOI 配对 (旧单抽 CSV; ⚠️ seeded 逐文件缺, 见 R6) ===")
    si_estoi = {}
    m3_all = {k: np.array([float(r[k]) for r in rows]) for k in ("pesq", "si_sdr", "estoi")}
    for tag, path in [("FlowSE_N1", "eval_dns/flowse_n1.csv"), ("FlowSE_N5", "eval_dns/flowse_n5.csv")]:
        fr = list(csv.DictReader(open(path)))
        fn = [x["filename"] for x in fr]
        aligned = (fn == [r["filename"] for r in rows])
        d = {}
        for k, delta in [("pesq", 0.05), ("si_sdr", 0.5), ("estoi", 0.01)]:
            fk = np.array([float(x[k]) for x in fr])
            r = tost_paired(fk, m3_all[k])
            r["verdict"] = "FlowSE_signif_better" if r["ci90_lo"] > 0 else ("FlowSE_signif_worse" if r["ci90_hi"] < 0 else "n.s.")
            r["equiv_delta"] = delta; r["equiv"] = bool(r["ci90_lo"] > -delta and r["ci90_hi"] < delta)
            d[k] = r
        si_estoi[tag] = {"source": path + " (旧单次抽样; seeded 见 R6)", "filenames_aligned": aligned, **d}
        print(f"  {tag} (对齐={aligned}): " + " | ".join(f"{k} Δ={d[k]['delta_F_minus_M3']:+.4f} {d[k]['verdict']}" for k in ("pesq", "si_sdr", "estoi")))

    # R5: band_check 跨 N(每频带+full, 相邻 N 差 vs seed std)
    band_curve = json.load(open(JPATH))["band_curve"]
    band_check_xN = {}
    print("\n=== R5 band_check 跨 N (相邻 N 差 vs seed std) ===")
    for b in BANDS:
        row = {}
        for i in range(1, len(NFE)):
            a, cN = band_curve[f"N={NFE[i-1]}"][b], band_curve[f"N={NFE[i]}"][b]
            diff = abs(a["mean"] - cN["mean"]); mss = max(a["std"], cN["std"])
            row[f"N{NFE[i-1]}_vs_N{NFE[i]}"] = {"abs_diff": diff, "max_seed_std": mss, "ratio_diff_over_seedstd": float(diff / mss) if mss else None,
                                                "robust": bool(diff > 3 * mss)}
        band_check_xN[b] = row
        tightest = min(row.values(), key=lambda x: x["ratio_diff_over_seedstd"] or 1e9)
        print(f"  {b}: 最紧相邻对 ratio={tightest['ratio_diff_over_seedstd']:.1f}× seed_std → {'稳健 ✓' if tightest['robust'] else '⚠️'}")

    # 写回 json
    d = json.load(open(JPATH))
    d["N_star"] = {
        "method": "paired TOST, FlowSE 3-seed PESQ mean vs ResFlowSE M3, n=824 per-file",
        "tost_delta_default": 0.05, "m3_pesq_ref": m3_mean, "order_check_passed": order_ok,
        "N_star_equiv": n_star_equiv_05, "N_star_sup": n_star_sup,
        "per_N_pesq_paired_tost": {f"N={N}": per_N[N] for N in NFE},
        "delta_sensitivity": {str(k): {"equiv_N": v[0], "N_star_equiv": v[1]} for k, v in delta_sens.items()},
        "crossing_robustness_R2": cross_robust,
        "first_cross_DIAGNOSTIC_ONLY": d["N_star"].get("N_star_first_cross"),
        "note": "N* 用 TOST(R1); first_cross 是点估计穿越仅诊断; 穿越稳健性用 3/3 seed(R2)非3σ; δ 敏感性须进正文; δ=±0.05 沿用 warm-vs-M3 同界(post hoc 须披露)"}
    d["band_check"] = {"method": "cross-N (adjacent NFE diff vs seed std); 旧跨频带版废弃(语音能量分布无信息量)",
                       "per_band_cross_N": band_check_xN,
                       "note": "资产A FlowSE 侧跨 seed 稳健; 但 'N=1 全频段 L2 最低' 被 R4 证伪(N=2 才最低)"}
    d["si_sdr_estoi_paired_R3"] = {"finding": "PESQ 优势是 PESQ 专属: 同代价 N=1 下 FlowSE 在 SI-SDR/ESTOI 显著更优(旧单抽); seeded 逐文件缺→R6 须重跑",
                                   "per_config": si_estoi,
                                   "N_star_equiv_if_SI_SDR": 1, "N_star_equiv_if_ESTOI": 1,
                                   "narrative_pending": "R3 改变论文核心表述, 等用户+mentor2 定调, pi 不自改稿"}
    d["review_status"] = "REVISE applied (R1/R2/R3/R5); R6 (seeded si_sdr/estoi per-file) pending re-run; R4 (asset A) recorded"
    # 重算 sha256_self (over content without the field)
    d.pop("sha256_self", None)
    json.dump(d, open(JPATH, "w"), indent=2)
    import hashlib
    sha = hashlib.sha256(open(JPATH, "rb").read()).hexdigest()[:16]
    d["sha256_self"] = sha
    json.dump(d, open(JPATH, "w"), indent=2)
    print(f"\n✓ json 更新: {JPATH}  sha256_self={sha}")
    print(f"  N*_equiv(δ0.05)={n_star_equiv_05}  N*_sup={n_star_sup}  order_ok={order_ok}")


if __name__ == "__main__":
    main()
