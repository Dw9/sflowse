"""T7q — FlowSE 质量 vs NFE, K=3 seed + 谱带MSE(REVIEW.md 末节裁定 + mentor2 第4旗)。
  - 固定seed保可复现 + K=3: PESQ/SI-SDR/ESTOI + per-band 复数谱MSE 均 mean±std over seeds
  - FlowSE 随机先验(odes.prior_sampling randn_like)故固定seed; ResFlowSE确定性免多抽样
  - N* = seed-平均PESQ曲线穿越 ResFlowSE PESQ; 邻域间距 <3×seed_std → 报区间不硬取整
  - 谱带MSE 档间差(low/mid/high) vs seed变异 核对(资产A FlowSE侧不能再建单次抽样)
reuse: eval_metrics.load_audio_16k / enhance_flowse(同 eval_dns 管线); band_mse 口径同 run_band_mse.py。"""
import argparse, os, json, numpy as np, torch
from glob import glob
from pesq import pesq
from pystoi import stoi
from flowmse.util.other import si_sdr
from flowmse.model import VFModel
from flowmse.data_module import SpecsDataModule
from eval_metrics import load_audio_16k, enhance_flowse
import torch.serialization
try:
    torch.serialization.add_safe_globals([SpecsDataModule])
except AttributeError:
    pass
SR = 16000
N_FFT, HOP = 510, 128
BANDS = {"low (0-1k)": (0, 32), "mid (1-4k)": (32, 128), "high (4-8k)": (128, 256)}


def stft_complex(wav_np, device, window):
    t = torch.from_numpy(np.ascontiguousarray(wav_np)).float().to(device)
    return torch.stft(t, n_fft=N_FFT, hop_length=HOP, window=window.to(device), center=True, return_complex=True)


def band_mse(S_enh, S_clean):
    d = S_enh - S_clean; cplx_sq = d.real ** 2 + d.imag ** 2
    out = {name: float(cplx_sq[lo:hi].mean()) for name, (lo, hi) in BANDS.items()}
    out["full"] = float(cplx_sq.mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="VB_DMD_FLOWSE_ICASSP_2025.ckpt")
    ap.add_argument("--data_dir", default="/home/zhibo/workspace/VoiceBank_processed")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--nfe", default="1,2,3,4,5")
    ap.add_argument("--resflowse_pesq_ref", default=None)
    ap.add_argument("--out_json", default="eval_dns/flowse_nfe_seeded.json")
    ap.add_argument("--csv_dir", default="eval_dns")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]; nfe = sorted(set(int(n) for n in args.nfe.split(",")))
    dev = torch.device(args.device)
    window = torch.hann_window(N_FFT, periodic=True)

    rf_ref = float(args.resflowse_pesq_ref) if args.resflowse_pesq_ref else None
    if rf_ref is None:
        try:
            rf_ref = float(json.load(open("eval_dns/m3.json"))["metrics"]["pesq"]["mean"])
        except Exception:
            rf_ref = 3.0622
    print(f"[T7q+bandMSE] 824 pairs; seeds={seeds} nfe={nfe} ResFlowSE_ref_PESQ={rf_ref:.4f}", flush=True)

    clean = sorted(glob(os.path.join(args.data_dir, "test", "clean", "*.wav")))
    noisy = sorted(glob(os.path.join(args.data_dir, "test", "noisy", "*.wav")))
    assert len(clean) == len(noisy) == 824, f"n={len(clean)}/{len(noisy)}"

    fse = VFModel.load_from_checkpoint(args.ckpt, base_dir=args.data_dir, map_location="cpu")
    try:
        fse.data_module.setup(stage=None)
    except Exception:
        pass
    for n, p in fse.dnn.named_parameters():
        if n in fse.ema_dnn:
            p.data = fse.ema_dnn[n].to(p.device)
    fse.eval(); fse.to(dev)

    per_seed = {}
    band_keys = list(BANDS) + ["full"]
    for seed in seeds:
        per_seed[seed] = {}
        for N in nfe:
            torch.manual_seed(seed)
            pesq_l, sisd_l, estoi_l = [], [], []
            band_sum = {k: 0.0 for k in band_keys}
            for i, (cf, nf) in enumerate(zip(clean, noisy)):
                x = load_audio_16k(cf); y = load_audio_16k(nf); T = x.size(1)
                xh = enhance_flowse(fse, y, T, N=N)
                xn = x.squeeze().numpy(); m = min(len(xn), len(xh)); xn, xh = xn[:m], xh[:m]
                pesq_l.append(pesq(SR, xn, xh, "wb")); sisd_l.append(si_sdr(xn, xh)); estoi_l.append(stoi(xn, xh, SR, extended=True))
                S_enh = stft_complex(xh, dev, window); S_cln = stft_complex(xn, dev, window)
                tt = min(S_enh.shape[1], S_cln.shape[1])
                bm = band_mse(S_enh[:, :tt], S_cln[:, :tt])
                for k in band_keys:
                    band_sum[k] += bm[k]
                if (i + 1) % 200 == 0:
                    print(f"  seed{seed} N={N} [{i+1}/824] PESQ~{np.mean(pesq_l):.4f}", flush=True)
            n824 = len(clean)
            per_seed[seed][N] = {"pesq": float(np.mean(pesq_l)), "si_sdr": float(np.mean(sisd_l)),
                                 "estoi": float(np.mean(estoi_l)),
                                 "band": {k: band_sum[k] / n824 for k in band_keys}}
            np.savetxt(os.path.join(args.csv_dir, f"flowse_n{N}_s{seed}_pesq.csv"), np.array(pesq_l), delimiter=",")
            print(f"  ✓ seed{seed} N={N}: PESQ={np.mean(pesq_l):.4f} SI-SDR={np.mean(sisd_l):.3f} ESTOI={np.mean(estoi_l):.4f} "
                  f"| band cplx low/mid/high/full = {band_sum['low (0-1k)']/n824:.5g}/{band_sum['mid (1-4k)']/n824:.5g}/{band_sum['high (4-8k)']/n824:.5g}/{band_sum['full']/n824:.5g}", flush=True)

    # 聚合 mean±std over seeds
    curve = {}; band_curve = {}
    for N in nfe:
        p = [per_seed[s][N]["pesq"] for s in seeds]; si = [per_seed[s][N]["si_sdr"] for s in seeds]; es = [per_seed[s][N]["estoi"] for s in seeds]
        curve[f"N={N}"] = {"pesq_mean": float(np.mean(p)), "pesq_std": float(np.std(p)),
                           "si_sdr_mean": float(np.mean(si)), "estoi_mean": float(np.mean(es)), "per_seed_pesq": [round(v, 5) for v in p]}
        bc = {}
        for k in band_keys:
            vals = [per_seed[s][N]["band"][k] for s in seeds]
            bc[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        band_curve[f"N={N}"] = bc

    # N*: seed-平均PESQ曲线穿越 rf_ref
    pms = [curve[f"N={n}"]["pesq_mean"] for n in nfe]; pss = [curve[f"N={n}"]["pesq_std"] for n in nfe]
    nstar = None
    if pms[0] >= rf_ref:
        nstar = {"call": "N=1 已≥ResFlowSE → FlowSE 截断即不劣; N*≤1", "N1_pesq": pms[0]}
    else:
        for i in range(1, len(nfe)):
            if pms[i] >= rf_ref:
                gap = pms[i] - pms[i-1]; thr = 3 * max(pss[i], pss[i-1])
                nstar = {"N_star_first_cross": nfe[i], "pesq_at_cross": pms[i], "prev_N": nfe[i-1], "prev_pesq": pms[i-1],
                         "neighborhood_gap": gap, "3x_seed_std": thr, "gap_lt_3std": bool(gap < thr),
                         "call": (f"N*∈{{{nfe[i-1]},{nfe[i]}}} 报区间" if gap < thr else f"N*={nfe[i]}")}
                break
        if nstar is None:
            nstar = {"call": f"FlowSE 在测程内(N≤{nfe[-1]})未追平 ResFlowSE({rf_ref:.4f}); N*>{nfe[-1]}", "max_pesq": pms[-1]}

    # 谱带MSE 档间差 vs seed变异 核对(mentor2 第4旗; 资产A 完整性)
    band_check = {}
    for N in nfe:
        bc = band_curve[f"N={N}"]
        # 档间差(相邻band mean 差) vs 该band的 seed std
        diffs = {"low_vs_mid": bc["low (0-1k)"]["mean"] - bc["mid (1-4k)"]["mean"],
                 "mid_vs_high": bc["mid (1-4k)"]["mean"] - bc["high (4-8k)"]["mean"]}
        max_seed_std = max(bc[k]["std"] for k in BANDS)
        band_check[f"N={N}"] = {"band_diffs": diffs, "max_band_seed_std": max_seed_std,
                                "diff_gt_3seedstd": {k: bool(abs(v) > 3 * max_seed_std) for k, v in diffs.items()},
                                "note": "档间差 > 3×seed_std → 该 N 的频带结构差异可信(非抽样噪声)"}

    sanity = None
    try:
        old = float(json.load(open("eval_dns/flowse_n1.json"))["metrics"]["pesq"]["mean"])
        s1 = curve["N=1"]["pesq_mean"]
        sanity = {"old_n1_2_8918": old, "seeded_N1_mean": s1, "abs_diff": abs(s1 - old), "within_floor_5e3": bool(abs(s1 - old) < 5e-3)}
    except Exception as e:
        sanity = str(e)

    out = {"task": "T7q FlowSE quality+NFE bandMSE (K=3 seed)", "seeds": seeds, "nfe": nfe, "resflowse_pesq_ref": rf_ref,
           "curve": curve, "band_curve": band_curve, "N_star": nstar, "band_check": band_check, "sanity_N1_vs_old": sanity,
           "note": "整条曲线 seed-averaged(PESQ+bandMSE); 资产A FlowSE侧不再单次抽样; N*邻域<3×seed_std报区间"}
    json.dump(out, open(args.out_json, "w"), indent=2)
    print("\n✓ saved " + args.out_json, flush=True)
    print(json.dumps({"N_star": nstar, "sanity": sanity,
                      "N1_band": band_curve.get("N=1"), "N5_band": band_curve.get("N=5")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
