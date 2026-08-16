"""p4_precheck.py — P4 前置诊断(REVIEW_P2 附3 派活1): 25W 功耗上限生效延迟假说检验。
协议: tegrastats 连采全程(100ms), 跑 ResFlowSE 全824 一遍(与 v5 主基准同负载同顺序, 无预热前置——
复现 v5 的首循环条件)。输出: 每分钟窗的 GPU 频率均值 / VDD_IN 均值 / RTF 均值 时间序列 +
首5min vs 稳态段(最后5min)对比。若频率/功率前几分钟高位后回落 → 假说确认(受限档测量须加预热至稳态);
全程恒定 → 假说证伪(P4 继续暂停报导师)。"""
import argparse, json, os, time, subprocess, re, threading, hashlib
import torch, numpy as np, soundfile as sf, torchaudio
from glob import glob
os.environ['PATH'] = os.path.expanduser('~/.local/bin') + ':/usr/local/cuda/bin:' + os.environ['PATH']
SR = 16000
SUDO_PW = os.environ.get("JETSON_SUDO_PW", "nx")


def load_audio(fp, sr=SR):
    info = sf.info(fp); dur = info.frames / info.samplerate
    y, sr0 = sf.read(fp); y = torch.from_numpy(y).float()
    if y.dim() == 1: y = y.unsqueeze(0)
    if sr0 != sr: y = torchaudio.functional.resample(y, sr0, sr)
    return y, dur


def prep_Y(model, y_wav, device):
    from flowmse.util.other import pad_spec
    yn = y_wav / y_wav.abs().max()
    return pad_spec(torch.unsqueeze(model._forward_transform(model._stft(yn.to(device))), 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noisy_dir", default=os.path.expanduser("~/sflowse_pkg/data/test/noisy"))
    ap.add_argument("--ckpt", default=os.path.expanduser("~/sflowse.ckpt"))
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    device = "cuda"
    from flowmse.resflowse_model import ResFlowSEModel
    noisy = sorted(glob(os.path.join(args.noisy_dir, "*.wav")))
    audios = [load_audio(f) for f in noisy]
    durs = np.array([d for _, d in audios])
    q = os.popen("nvpmodel -q").read().strip().splitlines()[-1]
    print(f"[P4-precheck] n={len(audios)} power={q}(须25W=3) — 无预热前置, 复现 v5 首循环条件", flush=True)

    # tegrastats 后台连采(100ms)
    samples = []  # (ts, vdd, gpu_freq, temp)
    proc = subprocess.Popen(f"echo {SUDO_PW} | sudo -S tegrastats --interval 100",
                            shell=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    def reader():
        for line in proc.stdout:
            ts = time.time()
            v = re.search(r'VDD_IN (\d+)mW', line)
            g = re.search(r'gpu@([\d.]+)C', line)
            f = re.search(r'GR3D_FREQ \d+%@\[(\d+)\]', line)
            samples.append((ts, int(v.group(1)) if v else None,
                            int(f.group(1)) if f else None, float(g.group(1)) if g else None))
    th = threading.Thread(target=reader, daemon=True); th.start()
    idle0 = time.time(); time.sleep(10)   # 10s idle 基线(冷启动)
    print("  idle 基线 10s 完成, 开始全824 首循环(无预热)", flush=True)

    rfs = ResFlowSEModel.load_from_checkpoint(args.ckpt, map_location="cpu", weights_only=False, strict=False)
    rfs.cuda().eval()
    # 模型 load 后 smoke 1 句(否则全崩, 不算预热)
    y0, _ = audios[0]; Y0 = prep_Y(rfs, y0, device)
    with torch.no_grad():
        xh = rfs.forward(Y0); _ = rfs.to_audio(xh.squeeze(), y0.size(1))
    torch.cuda.synchronize()

    lat = []; ts_log = []
    t0 = time.time()
    for i, (y, dur) in enumerate(audios):
        T = y.size(1); Y = prep_Y(rfs, y, device)
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        ts_ = time.time()
        s.record()
        with torch.no_grad():
            xh = rfs.forward(Y); _ = rfs.to_audio(xh.squeeze(), T)
        e.record(); torch.cuda.synchronize()
        lat.append(s.elapsed_time(e)); ts_log.append(ts_)
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/824] lat_mean {np.mean(lat):.1f}ms elapsed {time.time()-t0:.0f}s", flush=True)
    time.sleep(5)   # 尾部 idle
    proc.terminate(); th.join(timeout=2)
    lat = np.array(lat); rtf = lat / 1000 / durs

    # 每分钟窗聚合(频率/功率/RTF)
    t0b = ts_log[0]
    bins = {}
    for ts, l, r in zip(ts_log, lat, rtf):
        bins.setdefault(int((ts - t0b) // 60), []).append(("rtf", r))
    for ts, vdd, fq, tp in samples:
        if ts >= t0b - 10:
            b = int((ts - t0b) // 60)
            if vdd: bins.setdefault(b, []).append(("vdd", vdd))
            if fq: bins.setdefault(b, []).append(("freq", fq))
    permin = []
    for b in sorted(bins):
        r = [x[1] for x in bins[b] if x[0] == "rtf"]; v = [x[1] for x in bins[b] if x[0] == "vdd"]; f = [x[1] for x in bins[b] if x[0] == "freq"]
        permin.append({"minute": b, "n": len(r),
                       "rtf_mean": float(np.mean(r)) if r else None,
                       "vdd_mW_mean": float(np.mean(v)) if v else None,
                       "gpu_freq_MHz_mean": float(np.mean(f)) if f else None})
        row = permin[-1]
        print("  min%d: RTF %s | VDD %s | freq %s" % (b,
              f"{row['rtf_mean']:.5f}" if row['rtf_mean'] else "-",
              f"{row['vdd_mW_mean']:.0f}mW" if row['vdd_mW_mean'] else "-",
              f"{row['gpu_freq_MHz_mean']:.0f}MHz" if row['gpu_freq_MHz_mean'] else "-"), flush=True)
    # 首5min vs 末5min
    def w(seg): return [x for x in permin if x["minute"] in seg and x["rtf_mean"]]
    first = [x for x in permin if x["minute"] < 5 and x["rtf_mean"]]
    last = [x for x in permin if x["minute"] >= permin[-1]["minute"] - 4 and x["rtf_mean"] and x["minute"] >= 0]
    f_rtf = np.mean([x["rtf_mean"] for x in first]); l_rtf = np.mean([x["rtf_mean"] for x in last]) if last else f_rtf
    f_v = np.mean([x["vdd_mW_mean"] for x in first if x["vdd_mW_mean"]]); l_v = np.mean([x["vdd_mW_mean"] for x in last if x["vdd_mW_mean"]]) if any(x["vdd_mW_mean"] for x in last) else f_v
    f_f = np.mean([x["gpu_freq_MHz_mean"] for x in first if x["gpu_freq_MHz_mean"]]); l_f = np.mean([x["gpu_freq_MHz_mean"] for x in last if x["gpu_freq_MHz_mean"]]) if any(x["gpu_freq_MHz_mean"] for x in last) else f_f
    verdict = {
        "first5min": {"rtf": float(f_rtf), "vdd_mW": float(f_v), "freq_MHz": float(f_f)},
        "last5min": {"rtf": float(l_rtf), "vdd_mW": float(l_v), "freq_MHz": float(l_f)},
        "rtf_first_vs_last_pct": float((l_rtf / f_rtf - 1) * 100),
        "freq_first_vs_last_pct": float((l_f / f_f - 1) * 100) if f_f else None,
        "vdd_first_vs_last_pct": float((l_v / f_v - 1) * 100) if f_v else None,
    }
    # 判定: 频率/功率首段高位后回落(末段更低) → 假说确认
    verdict["hypothesis"] = ("SUPPORTED: 首5min频率/功率高于末5min(回落) → 功耗上限生效延迟, 受限档测量须加'预热至稳态'前置" 
                             if (verdict["freq_first_vs_last_pct"] or 0) > 2 or (verdict["vdd_first_vs_last_pct"] or 0) > 3
                             else "REFUTED: 全程恒定 → 功耗上限延迟假说不成立, P4 继续暂停另找根因")
    print("\n判定: " + verdict["hypothesis"], flush=True)
    out = {"task": "P4 前置: 25W 功耗上限生效延迟假说检验", "power": q, "n": len(audios),
           "rtf_overall_mean": float(rtf.mean()), "per_minute": permin, "verdict": verdict,
           "note": "无预热首循环(复现v5条件); 判据: 首5min vs 末5min 频率/功率差>2%/3%"}
    json.dump(out, open(args.output, "w"), indent=2)
    sha = hashlib.sha256(open(args.output, "rb").read()).hexdigest()
    open(args.output + ".sha256", "w").write(f"{sha}  {args.output}\n")
    print(f"✓ saved {args.output} sha={sha[:16]}", flush=True)


if __name__ == "__main__":
    main()
