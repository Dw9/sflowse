#!/usr/bin/env python3
"""First-file sensitivity of per-file RTF statistics (P1-1 of EDITORIAL_DECISION_v2).

Motivation (R1-M3/R2-8/DA-M13, three-panel convergence): the benchmark protocol logs
no dedicated warm-up rounds, so the first file of each pass can carry one-off CUDA
context / autotune cost that lands exactly in the tail that the N_max (p95) verdict
reads.  This script quantifies that effect from the retained per-file arrays:

  - bench_orin_{15w,10w}_v6_expB_perfile.npy : N=2, 824 x 3 passes (clean sessions)
  - maxn_ab_v3_perfile.npy                   : N=1, 824 x 1 (interleaved A/B, arm A)
  - bench_orin_25w_v5_expB_perfile.npy       : N=2, 824 x 3 (POLLUTED session -> side
                                                evidence only, flagged in output)

For each: pooled mean/p95 over all entries vs pooled stats after dropping the first
file of EVERY pass; plus the rank of each first file within its pass (0 = slowest).
MAXN N=5/N=6 full-824 arrays were not retained (json aggregates only) -> covered in
the manuscript by a rank argument, calibrated by the effect sizes measured here.

Percentile convention: np.percentile default (linear), same as bench_jetson.py.
Cross-check: recomputed p95_all is compared against the parent json's reported
experiment_B_load rtf p95 where available; a mismatch >1% fails the run (fail-closed).

Output: firstfile_sensitivity.json (+ sha256 of itself written to stdout for ledger).
"""
import hashlib
import json
import sys

import numpy as np

BENCH = 824  # files per pass

SOURCES = [
    dict(tag="15W_v6_N2", npy="bench_orin_15w_v6_expB_perfile.npy",
         json="bench_orin_15w_v6.json", polluted=False),
    dict(tag="10W_v6_N2", npy="bench_orin_10w_v6_expB_perfile.npy",
         json="bench_orin_10w_v6.json", polluted=False),
    dict(tag="MAXN_ab_v3_N1_armA", npy="maxn_ab_v3_perfile.npy",
         json=None, polluted=False),
    dict(tag="25W_v5_N2_POLLUTED", npy="bench_orin_25w_v5_expB_perfile.npy",
         json="bench_orin_25w_v5.json", polluted=True),
]


def pooled(rtf):
    return dict(n=int(rtf.size), mean=float(np.mean(rtf)), p95=float(np.percentile(rtf, 95)))


def drop_first_per_pass(rtf, n_passes):
    keep = [rtf[p * BENCH:(p + 1) * BENCH][1:] for p in range(n_passes)]
    return np.concatenate(keep)


def main():
    out = {"script": "analyze_firstfile_p95.py", "bench_files_per_pass": BENCH,
           "percentile": "np.percentile linear",
           "rank_convention": ("rank_from_slowest_within_pass counts, inside one pass of "
                               f"{BENCH} files, the files strictly slower than that pass's "
                               "first file; pooled-percentile equivalence: within-pass rank/N "
                               "equals pooled rank/(N*passes) since passes are iid draws"),
           "sources": []}
    for spec in SOURCES:
        d = np.load(spec["npy"], allow_pickle=True).item()
        # arm A only for the interleaved A/B artifact (ResFlowSE single-step)
        rtf = np.asarray(d["rtfA"] if "rtfA" in d else d["rtf"], dtype=float)
        n = rtf.size
        if n % BENCH:
            sys.exit(f"[FAIL] {spec['tag']}: n={n} not a multiple of {BENCH}")
        n_passes = n // BENCH

        all_stats = pooled(rtf)
        dropped = pooled(drop_first_per_pass(rtf, n_passes))

        firsts = []
        for p in range(n_passes):
            seg = rtf[p * BENCH:(p + 1) * BENCH]
            f = seg[0]
            rank = int(np.sum(seg > f))  # within THIS pass: files strictly slower
            firsts.append(dict(pass_=p, rtf=float(f),
                               rank_from_slowest_within_pass=rank,
                               pct_from_slowest_within_pass=round(100.0 * rank / len(seg), 2),
                               pass_p95=float(np.percentile(seg, 95)),
                               first_over_pass_p95=float(f / np.percentile(seg, 95))))

        # fail-closed cross-check against the parent json aggregate, when present
        xcheck = None
        if spec["json"]:
            j = json.load(open(spec["json"]))
            ref = (j.get("experiment_B_load", {}).get("rtf", {}) or {}).get("p95")
            if ref is not None and abs(all_stats["p95"] / ref - 1) > 0.01:
                sys.exit(f"[FAIL] {spec['tag']}: recomputed p95 {all_stats['p95']:.5f} "
                         f"vs json {ref:.5f} (>1%)")
            xcheck = None if ref is None else dict(json_p95=ref,
                                                   recomputed_p95=round(all_stats["p95"], 5))

        out["sources"].append(dict(
            tag=spec["tag"], npy=spec["npy"], n_passes=n_passes, polluted=spec["polluted"],
            all=all_stats, drop_first=dropped,
            p95_shift_pct=round(100.0 * (dropped["p95"] / all_stats["p95"] - 1), 3),
            mean_shift_pct=round(100.0 * (dropped["mean"] / all_stats["mean"] - 1), 3),
            first_files=firsts, json_crosscheck=xcheck))

    with open("firstfile_sensitivity.json", "w") as f:
        json.dump(out, f, indent=1)
    sha = hashlib.sha256(open("firstfile_sensitivity.json", "rb").read()).hexdigest()[:16]
    print(json.dumps({s["tag"]: dict(p95_shift_pct=s["p95_shift_pct"],
                                     mean_shift_pct=s["mean_shift_pct"],
                                     polluted=s["polluted"]) for s in out["sources"]}, indent=1))
    print(f"[ok] firstfile_sensitivity.json sha256:{sha}")


if __name__ == "__main__":
    main()
