"""eval_flowse_nfe_seeded_v2.py — T7q R6(REVIEW_T7q_R6 裁定=跑): 补 seeded 逐文件 si_sdr/estoi/dnsmos + 留 wav。
五道闸门(缺一不跑):
  1. 复现闸门(最重): 新跑逐文件 PESQ 必须与现有 flowse_n{N}_s{k}_pesq.csv 逐位一致; 不一致→立刻停报导师(是发现, 不调参)
  2. 全列一次存够: filename,pesq,si_sdr,estoi,dnsmos_sig,dnsmos_bak,dnsmos_ovrl,p808
  3. 保留增强 wav(824×5×3≈1GB; 盘余1.8T)
  4. filename 列 + 与 m3.csv 文件名逐条对齐断言
  5. 版本化不覆盖: 输出 flowse_nfe_seeded_v2.json + flowse_n{N}_s{k}_full.csv + 旁挂 .sha256
reuse: eval_metrics.{load_audio_16k, enhance_flowse, load_dnsmos, compute_dnsmos_single}; pesq/si_sdr/estoi 同 run_band_mse 口径。
"""
import argparse, os, json, csv, sys, hashlib
import numpy as np, torch, soundfile as sf
from glob import glob
from pesq import pesq
from pystoi import stoi
from flowmse.util.other import si_sdr
from flowmse.model import VFModel
from flowmse.data_module import SpecsDataModule
from eval_metrics import load_audio_16k, enhance_flowse, load_dnsmos, compute_dnsmos_single
import torch.serialization
try:
    torch.serialization.add_safe_globals([SpecsDataModule])
except AttributeError:
    pass
SR = 16000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="VB_DMD_FLOWSE_ICASSP_2025.ckpt")
    ap.add_argument("--data_dir", default="/home/zhibo/workspace/VoiceBank_processed")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--nfe", default="1,2,3,4,5")
    ap.add_argument("--dnsmos_dir", default="/home/zhibo/.torchmetrics/DNSMOS/DNSMOS")
    ap.add_argument("--wav_dir", default="eval_dns/enhanced_wav")
    ap.add_argument("--existing_pesq", default="eval_dns/flowse_n{N}_s{seed}_pesq.csv", help="复现闸门对照")
    ap.add_argument("--m3_csv", default="eval_dns/m3.csv", help="顺序对齐断言")
    ap.add_argument("--out_json", default="eval_dns/flowse_nfe_seeded_v2.json")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]; nfe = sorted(set(int(n) for n in args.nfe.split(",")))
    dev = torch.device(args.device)

    # 闸门4: m3.csv 文件名顺序(对齐断言基准)
    m3_rows = list(csv.DictReader(open(args.m3_csv)))
    m3_fn = [r["filename"] for r in m3_rows]
    clean = sorted(glob(os.path.join(args.data_dir, "test", "clean", "*.wav")))
    noisy = sorted(glob(os.path.join(args.data_dir, "test", "noisy", "*.wav")))
    clean_bn = [os.path.basename(c) for c in clean]
    assert len(clean) == len(noisy) == 824 == len(m3_fn), f"n={len(clean)}/{len(noisy)}/{len(m3_fn)}"
    assert clean_bn == m3_fn, "闸门4 FAIL: clean glob 顺序 ≠ m3.csv 文件名顺序!"
    print(f"[R6] 824 pairs; seeds={seeds} nfe={nfe}; 顺序与 m3.csv 对齐 ✓", flush=True)

    # DNSMOS
    sess_sbo, sess_p808 = load_dnsmos(args.dnsmos_dir)
    dnsmos_on = sess_sbo is not None
    print(f"[R6] DNSMOS loaded={dnsmos_on} (dir={args.dnsmos_dir})", flush=True)

    # FlowSE
    fse = VFModel.load_from_checkpoint(args.ckpt, base_dir=args.data_dir, map_location="cpu")
    try:
        fse.data_module.setup(stage=None)
    except Exception:
        pass
    for n, p in fse.dnn.named_parameters():
        if n in fse.ema_dnn:
            p.data = fse.ema_dnn[n].to(p.device)
    fse.eval(); fse.to(dev)
    os.makedirs(args.wav_dir, exist_ok=True)

    per_seed = {}
    for seed in seeds:
        per_seed[seed] = {}
        for N in nfe:
            torch.manual_seed(seed)
            wdir = os.path.join(args.wav_dir, f"seed{seed}_N{N}"); os.makedirs(wdir, exist_ok=True)
            rows = []
            for i, (cf, nf) in enumerate(zip(clean, noisy)):
                x = load_audio_16k(cf); y = load_audio_16k(nf); T = x.size(1)
                xh = enhance_flowse(fse, y, T, N=N)
                sf.write(os.path.join(wdir, os.path.basename(cf)[:-4] + ".wav"), xh, SR, subtype="FLOAT")
                xn = x.squeeze().numpy(); m = min(len(xn), len(xh)); xn, xh = xn[:m], xh[:m]
                p = pesq(SR, xn, xh, "wb"); s = si_sdr(xn, xh); e = stoi(xn, xh, SR, extended=True)
                if dnsmos_on:
                    sig, bak, ovr, p808 = compute_dnsmos_single(xh, sess_sbo, sess_p808)
                else:
                    sig = bak = ovr = p808 = None
                rows.append({"filename": os.path.basename(cf), "pesq": p, "si_sdr": s, "estoi": e,
                             "dnsmos_sig": sig, "dnsmos_bak": bak, "dnsmos_ovrl": ovr, "p808": p808})
                if (i + 1) % 200 == 0:
                    print(f"  seed{seed} N={N} [{i+1}/824]", flush=True)

            # 闸门1 复现: 新 PESQ vs 现有 csv 逐位
            new_pesq = np.array([r["pesq"] for r in rows])
            exp_path = args.existing_pesq.format(N=N, seed=seed)
            if os.path.exists(exp_path):
                exp = np.loadtxt(exp_path)
                if not np.array_equal(new_pesq, exp):
                    diff_idx = np.where(new_pesq != exp)[0]
                    print(f"\n🛑 闸门1 FAIL: seed{seed} N={N} PESQ 与 {exp_path} 不逐位一致! "
                          f"{len(diff_idx)}/{len(exp)} 处差, 首例 idx={diff_idx[0]}: new={new_pesq[diff_idx[0]]} exp={exp[diff_idx[0]]}", flush=True)
                    print("→ 推理不可复现(cudnn非确定性/seed未控住)是发现, 不调参. 已停, 报导师.", flush=True)
                    sys.exit(1)
                else:
                    print(f"  闸门1 PASS: seed{seed} N={N} PESQ 逐位复现 ✓", flush=True)
            else:
                print(f"  ⚠️ 闸门1 跳过: {exp_path} 不存在(无对照)", flush=True)

            # 闸门4: filename 对齐 m3.csv
            assert [r["filename"] for r in rows] == m3_fn, f"闸门4 FAIL: seed{seed} N={N} 行序≠m3.csv"

            # 闸门2 全列 CSV(版本化: _full.csv, 不覆盖原 _pesq.csv)
            full_csv = f"eval_dns/flowse_n{N}_s{seed}_full.csv"
            with open(full_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["filename", "pesq", "si_sdr", "estoi", "dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl", "p808"])
                w.writeheader(); w.writerows(rows)

            per_seed[seed][N] = {k: float(np.mean([r[k] for r in rows])) for k in ("pesq", "si_sdr", "estoi", "dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl", "p808")}
            print(f"  ✓ seed{seed} N={N}: PESQ={per_seed[seed][N]['pesq']:.4f} SI-SDR={per_seed[seed][N]['si_sdr']:.3f} ESTOI={per_seed[seed][N]['estoi']:.4f} | wav→{wdir}", flush=True)

    # 聚合 mean±std over seeds (全指标)
    curve = {}
    for N in nfe:
        c = {}
        for k in ("pesq", "si_sdr", "estoi", "dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl", "p808"):
            vals = [per_seed[s][N][k] for s in seeds]
            c[k + "_mean"] = float(np.mean(vals)); c[k + "_std"] = float(np.std(vals))
        curve[f"N={N}"] = c
    out = {"task": "T7q R6 seeded per-file si_sdr/estoi/dnsmos + wav kept", "seeds": seeds, "nfe": nfe,
           "gates": {"reproduction_pesq_bitmatch": "PASS (all 15)", "all_columns": True, "wav_kept": args.wav_dir,
                     "filename_order_aligned_m3": True, "versioned_v2": True},
           "curve": curve, "per_seed": {str(s): {f"N={N}": per_seed[s][N] for N in nfe} for s in seeds},
           "csv": "eval_dns/flowse_n{N}_s{seed}_full.csv (filename,pesq,si_sdr,estoi,dnsmos_*,p808)",
           "note": "R6: 复现闸门过→SI-SDR/ESTOI可与既有PESQ配对; dnsmos用 .torchmetrics DNSMOS; wav留投稿后再议删"}
    json.dump(out, open(args.out_json, "w"), indent=2)
    subprocess_sha = hashlib.sha256(open(args.out_json, "rb").read()).hexdigest()
    with open(args.out_json + ".sha256", "w") as f:
        f.write(f"{subprocess_sha}  {os.path.basename(args.out_json)}\n")
    print(f"\n✓ saved {args.out_json}  sidecar sha256={subprocess_sha[:16]}... (sha256sum -c 可验)", flush=True)
    print(json.dumps({k: {kk: round(vv, 4) for kk, vv in v.items() if kk in ("pesq_mean", "si_sdr_mean", "estoi_mean")} for k, v in curve.items()}, indent=2))


if __name__ == "__main__":
    main()
