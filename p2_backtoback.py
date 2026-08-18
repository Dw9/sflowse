"""p2_backtoback.py — P2.3 背靠背复测 + P2 裁定必做1(N=3 全824), 25W, 单一顺序:
  warmup(两模型充分同预热) → ResFlowSE 单步 全824 → FlowSE N=1 全824 → (差值即口径诊断) → FlowSE N=3 全824
同会话/同预热/同顺序, cuda.Event, caliber 同 bench_jetson v5. 输出版本化 json + 旁挂 sha。"""
import argparse, json, os, time, hashlib
import torch, numpy as np, soundfile as sf, torchaudio
from glob import glob
os.environ['PATH'] = os.path.expanduser('~/.local/bin') + ':/usr/local/cuda/bin:' + os.environ['PATH']   # ninja(NCSN++ CUDA op JIT 必需)
SR = 16000
CALIBER = ("cuda.Event(network_forward + iSTFT); Y precomputed; batch=1; 48k->resample16k; "
           "back-to-back same-session same-warmup ResFlowSE_single -> FlowSE_N1 -> FlowSE_N3")


def load_audio(fp, sr=SR):
    info = sf.info(fp); dur = info.frames / info.samplerate
    y, sr0 = sf.read(fp); y = torch.from_numpy(y).float()
    if y.dim() == 1: y = y.unsqueeze(0)
    if sr0 != sr: y = torchaudio.functional.resample(y, sr0, sr)
    return y, dur


def prep_Y(model, y_wav, device):
    from flowmse.util.other import pad_spec
    yn = y_wav / y_wav.abs().max()
    Y = torch.unsqueeze(model._forward_transform(model._stft(yn.to(device))), 0)
    return pad_spec(Y)


def time_fwd(fn):
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record(); fn(); e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e)


def stats(a):
    a = np.asarray(a, float)
    return {"n": int(len(a)), "mean": float(a.mean()), "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)), "p99": float(np.percentile(a, 99)), "std": float(a.std())}


def run_pass(model_kind, fn_for, audios, device, tag):
    lat = []
    for y, _ in audios:
        T = y.size(1); Y = prep_Y(fn_for["model"], y, device)
        lat.append(time_fwd(fn_for["fn"](Y, T)))
    lat = np.array(lat)
    durs = np.array([d for _, d in audios])
    rtf = lat / 1000 / durs
    print(f"  [{tag}] lat mean {lat.mean():.2f}ms | RTF mean {rtf.mean():.5f} p95 {np.percentile(rtf,95):.5f}", flush=True)
    return {"latency_ms": stats(lat), "rtf": stats(rtf)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noisy_dir", default=os.path.expanduser("~/sflowse_pkg/data/test/noisy"))
    ap.add_argument("--resflowse_ckpt", default=os.path.expanduser("~/sflowse.ckpt"))
    ap.add_argument("--flowse_ckpt", default=os.path.expanduser("~/VB_DMD_FLOWSE_ICASSP_2025.ckpt"))
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    device = "cuda"
    from flowmse.resflowse_model import ResFlowSEModel
    from flowmse.data_module import SpecsDataModule
    from flowmse.model import VFModel
    from flowmse.sampling import get_white_box_solver
    import torch.serialization
    try: torch.serialization.add_safe_globals([SpecsDataModule])
    except AttributeError: pass

    noisy = sorted(glob(os.path.join(args.noisy_dir, "*.wav")))
    audios = [load_audio(f) for f in noisy]
    print(f"[back-to-back 25W] n={len(audios)}", flush=True)
    q = os.popen("nvpmodel -q").read().strip().splitlines()[-1]
    print("  power:", q, flush=True)

    rfs = ResFlowSEModel.load_from_checkpoint(args.resflowse_ckpt, map_location="cpu", weights_only=False, strict=False)
    rfs.cuda().eval()
    fse = VFModel.load_from_checkpoint(args.flowse_ckpt, base_dir=os.path.dirname(os.path.dirname(args.noisy_dir.rstrip("/"))), map_location="cpu")
    try: fse.data_module.setup(stage=None)
    except Exception: pass
    for n, p in fse.dnn.named_parameters():
        if n in fse.ema_dnn: p.data = fse.ema_dnn[n].to(p.device)
    fse.cuda().eval()

    def rfs_one(Y, T):
        def _f():
            with torch.no_grad():
                xh = rfs.forward(Y); _ = rfs.to_audio(xh.squeeze(), T)
        return _f

    def fse_one(Y, T, N=1):
        def _f():
            with torch.no_grad():
                sampler = get_white_box_solver("euler", fse.ode, fse, Y, T_rev=fse.T_rev, t_eps=fse.t_eps, N=N)
                sample, _ = sampler(); _ = fse.to_audio(sample.squeeze(), T)
        return _f

    # 充分同预热: 两模型各跑 20 句交错
    print("  warmup: 两模型交错 20 句...", flush=True)
    for y, _ in audios[:20]:
        Yr = prep_Y(rfs, y, device); rfs_one(Yr, y.size(1))()
        Yf = prep_Y(fse, y, device); fse_one(Yf, y.size(1), 1)()
    torch.cuda.synchronize()

    out = {"caliber": CALIBER, "power": q, "n": len(audios), "order": ["ResFlowSE_single", "FlowSE_N1", "FlowSE_N3"]}
    r = run_pass("rfs", {"model": rfs, "fn": rfs_one}, audios, device, "ResFlowSE 单步 全824")
    f1 = run_pass("fse", {"model": fse, "fn": lambda Y, T: fse_one(Y, T, 1)}, audios, device, "FlowSE N=1 全824")
    diff_pct = (f1["rtf"]["mean"] - r["rtf"]["mean"]) / r["rtf"]["mean"] * 100
    print(f"  差值: FlowSE_N1 - ResFlowSE = +{diff_pct:.1f}% (P2.3 异常复核: 此前全824差 15%, MAXN 1.7%)", flush=True)
    f3 = run_pass("fse", {"model": fse, "fn": lambda Y, T: fse_one(Y, T, 3)}, audios, device, "FlowSE N=3 全824")
    out["resflowse_single"] = r; out["flowse_N1"] = f1; out["flowse_N3"] = f3
    out["N1_vs_single_diff_pct"] = diff_pct
    out["N3_p95_ge_1"] = bool(f3["rtf"]["p95"] >= 1.0)
    json.dump(out, open(args.output, "w"), indent=2)
    sha = hashlib.sha256(open(args.output, "rb").read()).hexdigest()
    open(args.output + ".sha256", "w").write(f"{sha}  {args.output}\n")
    print(f"\n✓ saved {args.output} sha={sha[:16]}", flush=True)
    print(json.dumps({"rfs_rtf_mean": r["rtf"]["mean"], "flowse_N1_rtf_mean": f1["rtf"]["mean"],
                      "diff_pct": diff_pct, "N3_rtf_mean": f3["rtf"]["mean"], "N3_p95": f3["rtf"]["p95"],
                      "N3_p95_ge_1": out["N3_p95_ge_1"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
