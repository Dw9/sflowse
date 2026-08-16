"""T2a — A100 torch fp32 延迟/内存基准 v2 (含 per-file RTF 分布).
口径(json caliber): network_forward + iSTFT(to_audio), 输入 STFT 预计算, batch=1.
计时: cuda.Event; 真实语料每句1次; 定长 warmup20/repeat200.
dur 用 soundfile.info 实际 sr 算(不硬编码 16000! —— 上版 RTF 错 3 倍的根因).
per-file RTF = latency_i / dur_i; 其分布(mean/p50/p95/p99/std)用于谈尾部(裸延迟 p99 被句长支配).
NFE 扫描同进程同热态 + 自洽检查.
"""
import argparse, json, os, torch, numpy as np, soundfile as sf, torchaudio
import torch.serialization  # module-level (avoid closure free-var bug)
from glob import glob

SR = 16000
CALIBER = ("latency = network_forward(model.forward or euler-solver NFE) + iSTFT(to_audio); "
           "input complex spectrogram Y precomputed (STFT+spec_fwd+pad_spec outside timing); batch=1; single utterance")


def load_audio(fp, sr=SR):
    info = sf.info(fp)
    dur_real = info.frames / info.samplerate  # 实际 sr 算时长(不硬编码!)
    y, sr0 = sf.read(fp); y = torch.from_numpy(y).float()
    if y.dim() == 1: y = y.unsqueeze(0)
    if sr0 != sr: y = torchaudio.functional.resample(y, sr0, sr)
    return y, dur_real


def prep_Y(model, y_wav, device):
    from flowmse.util.other import pad_spec
    norm = y_wav.abs().max(); yn = y_wav / norm
    Y = torch.unsqueeze(model._forward_transform(model._stft(yn.to(device))), 0)
    return pad_spec(Y), norm


def stats(a):
    a = np.asarray(a, float)
    return {"n": int(len(a)), "mean": float(a.mean()), "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)), "p99": float(np.percentile(a, 99)),
            "std": float(a.std())}


def time_ev(fn, n_warmup, n_repeat):
    for _ in range(n_warmup): fn()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(n_repeat):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return np.array(times), torch.cuda.max_memory_allocated()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resflowse_ckpt", default="sflowse.ckpt")
    ap.add_argument("--flowse_ckpt", default="VB_DMD_FLOWSE_ICASSP_2025.ckpt")
    ap.add_argument("--data_dir", default="/home/zhibo/workspace/VoiceBank_processed")
    ap.add_argument("--n_real", type=int, default=824)
    ap.add_argument("--output", default="bench_a100.json")
    args = ap.parse_args()
    device = "cuda"
    from flowmse.util.other import pad_spec
    from flowmse.resflowse_model import ResFlowSEModel

    noisy = sorted(glob(os.path.join(args.data_dir, "test", "noisy", "*.wav")))
    n_real = min(args.n_real, len(noisy)); noisy = noisy[:n_real]
    # 预读所有 (y, dur_real)
    audios = [load_audio(nf) for nf in noisy]
    durs = np.array([d for _, d in audios])
    print(f"[bench_a100 v2] A100 fp32, real={n_real}, mean_dur={durs.mean():.4f}s sr=48k(resample16k)")

    rfs = ResFlowSEModel.load_from_checkpoint(args.resflowse_ckpt, map_location="cpu", weights_only=False, strict=False)
    rfs.cuda().eval()

    def rfs_fn(Y, T): 
        def _f():
            with torch.no_grad():
                xh = rfs.forward(Y); _ = rfs.to_audio(xh.squeeze(), T)
        return _f

    # warmup
    for y, _ in audios[:5]:
        Y, _ = prep_Y(rfs, y, device); rfs_fn(Y, y.size(1))()
    torch.cuda.synchronize()
    # 真实语料 per-file
    rfs_lat = []
    for y, _ in audios:
        T = y.size(1); Y, _ = prep_Y(rfs, y, device)
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); rfs_fn(Y, T)(); e.record(); torch.cuda.synchronize()
        rfs_lat.append(s.elapsed_time(e))
    rfs_lat = np.array(rfs_lat)
    rfs_rtf = rfs_lat / 1000 / durs
    rfs_real = {"latency_ms": stats(rfs_lat), "rtf": stats(rfs_rtf), "mean_dur_s": float(durs.mean())}
    print(f"  ResFlowSE: lat mean={rfs_lat.mean():.2f}ms RTF mean={rfs_rtf.mean():.5f} p95={np.percentile(rfs_rtf,95):.5f}")

    # 定长
    base = audios[len(audios)//2][0]
    fixed = {}
    for L in [1, 2, 4, 8, 16]:
        ns = L * SR
        w = base[:, :ns] if base.size(1) >= ns else base.repeat(1, (ns+base.size(1)-1)//base.size(1))[:, :ns]
        Y, _ = prep_Y(rfs, w, device)
        ts, peak = time_ev(rfs_fn(Y, L*SR), 20, 200)
        fixed[f"{L}s"] = {"latency_ms": stats(ts), "rtf": float(ts.mean()/1000/L), "peak_mem_bytes": int(peak)}
    torch.cuda.empty_cache()

    # FlowSE NFE
    from flowmse.data_module import SpecsDataModule
    from flowmse.model import VFModel
    from flowmse.sampling import get_white_box_solver
    torch.serialization.add_safe_globals([SpecsDataModule])
    fse = VFModel.load_from_checkpoint(args.flowse_ckpt, base_dir=args.data_dir, map_location="cpu")
    fse.data_module.setup(stage=None)
    for name, p in fse.dnn.named_parameters():
        if name in fse.ema_dnn: p.data = fse.ema_dnn[name].to(p.device)
    fse.cuda().eval()
    T_rev, t_eps = fse.T_rev, fse.t_eps

    def fse_fn(Y, T, N):
        def _f():
            with torch.no_grad():
                sampler = get_white_box_solver("euler", fse.ode, fse, Y, T_rev=T_rev, t_eps=t_eps, N=N)
                sample, _ = sampler(); _ = fse.to_audio(sample.squeeze(), T)
        return _f

    fse_real = {}
    fse_lat_byN = {}
    for N in [1, 2, 3, 5]:
        for y, _ in audios[:3]:
            Y, _ = prep_Y(fse, y, device); fse_fn(Y, y.size(1), N)()
        torch.cuda.synchronize()
        lat = []
        for y, _ in audios:
            T = y.size(1); Y, _ = prep_Y(fse, y, device)
            s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
            s.record(); fse_fn(Y, T, N)(); e.record(); torch.cuda.synchronize()
            lat.append(s.elapsed_time(e))
        lat = np.array(lat)
        fse_lat_byN[N] = lat
        fse_real[f"N={N}"] = {"latency_ms": stats(lat), "rtf": stats(lat/1000/durs)}
        print(f"  FlowSE N={N}: lat mean={lat.mean():.2f}ms RTF mean={(lat/1000/durs).mean():.5f}")

    rfs_m = rfs_lat.mean(); fN1 = fse_lat_byN[1].mean(); fN5 = fse_lat_byN[5].mean()
    per_step = fse_lat_byN[2].mean() - fN1
    consistency = {
        "i_resflowse_vs_flowse_N1": {"resflowse_ms": rfs_m, "flowse_N1_ms": fN1,
                                     "rel_diff_pct": abs(rfs_m-fN1)/max(rfs_m,fN1)*100},
        "ii_N5_linear": {"per_step_N2_N1": per_step, "per_step_N5_N1_over4": (fN5-fN1)/4,
                         "linear_check_ok": bool(abs(per_step-(fN5-fN1)/4)/max(per_step,1e-9) < 0.15)},
    }

    out = {
        "caliber": CALIBER,
        "platform": {"device": torch.cuda.get_device_name(0), "torch": torch.__version__, "precision": "fp32",
                     "tf32": {"matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                              "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32)},
                     "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "?")},
        "audio": {"sr_hz": 48000, "mean_dur_s": float(durs.mean()), "median_dur_s": float(np.median(durs)),
                  "min_dur_s": float(durs.min()), "max_dur_s": float(durs.max()),
                  "note": "wav实存48kHz(eval resample 16k); dur用sf.info实际sr算"},
        "resflowse_1nfe": {"real_824": rfs_real, "fixed_len": fixed, "per_file": {
            "latency_ms": rfs_lat.tolist(), "dur_s": durs.tolist(), "rtf": rfs_rtf.tolist()}, "ckpt": args.resflowse_ckpt},
        "flowse_nfe_sweep": {"real_824": fse_real, "ckpt": args.flowse_ckpt},
        "rtf_summary": {"resflowse_1nfe": rfs_real["rtf"]["mean"], "flowse_N1": fse_real["N=1"]["rtf"]["mean"],
                        "flowse_N5": fse_real["N=5"]["rtf"]["mean"], "speedup_N5_over_1nfe": fN5/rfs_m,
                        "method": "per-file RTF=latency_i/dur_i, then stats"},
        "decomposition": {"per_step_forward_ms": per_step, "iSTFT_ms_approx": fN1 - per_step,
                          "note": "iSTFT=N1-(N2-N1); 延迟几乎全在网络前向→缩容唯一杠杆"},
        "nfe_sweep": {"consistency_check": consistency},
        "note": "v2 含 per-file RTF 分布(谈尾部用 per-file RTF, 非裸延迟 p99)",
    }
    json.dump(out, open(args.output, "w"), indent=2)
    print(f"\n✓ saved {args.output}")
    print(json.dumps({"rtf_summary": out["rtf_summary"], "consistency": consistency,
                      "rfs_rtf_p95": rfs_real["rtf"]["p95"], "rfs_rtf_p99": rfs_real["rtf"]["p99"]}, indent=2))


if __name__ == "__main__":
    main()
