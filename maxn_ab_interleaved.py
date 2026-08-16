"""p2_ab_interleaved.py — M24.4 逐文件交错 A/B(消位置效应, 替代块设计背靠背; REVIEW_P2 派活1)
协议: 全 824 文件, 每文件依次跑 ResFlowSE单步(A) 与 FlowSE N=1(B), 顺序交替 A/B/B/A/A/B/B/A...(4周期块),
消位置/热/缓存漂移混淆。同会话同预热。输出: 逐文件配对差 + mean/p95 + 配对 t + 逐句 npy。
MAXN 版: 只测 A/B 交错(派活3, 复测 0.16523)。"""
import argparse, json, os, time, hashlib
import torch, numpy as np, soundfile as sf, torchaudio
from glob import glob
import scipy.stats as sstats   # ⚠️ 别叫 stats——与自定 stats() 遮蔽(上次就这么崩的)
os.environ['PATH'] = os.path.expanduser('~/.local/bin') + ':/usr/local/cuda/bin:' + os.environ['PATH']
SR = 16000

def env_snapshot(tag):
    """环境快照(永久协议, 用户裁定 2026-08-15): load + ps top10 + 监控面板在否, 写进每次测量 json 前后各一。"""
    import subprocess
    def sh(c):
        try: return subprocess.run(c, shell=True, capture_output=True, text=True, timeout=15).stdout
        except Exception as e: return f"<{e}>"
    ps = sh("ps aux --sort=-%cpu | head -11")
    return {"tag": tag, "at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "load_1_5_15": open("/proc/loadavg").read().split()[:3],
            "ps_top10_cpu": ps,
            "jtop_running": "jtop" in ps, "update_manager_running": "update-manager" in ps}

CALIBER = ("cuda.Event(network_forward + iSTFT); batch=1; 48k->resample16k; "
           "per-file interleaved A(ResFlowSE_single)/B(FlowSE_N1) with ABBA rotation")


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


def time_fwd(fn):
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record(); fn(); e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e)


def stats(a):
    a = np.asarray(a, float)
    return {"n": int(len(a)), "mean": float(a.mean()), "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)), "p99": float(np.percentile(a, 99)), "std": float(a.std())}


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

    ENV_BEFORE = env_snapshot("before")
    noisy = sorted(glob(os.path.join(args.noisy_dir, "*.wav")))
    audios = [load_audio(f) for f in noisy]
    durs = np.array([d for _, d in audios])
    q = os.popen("nvpmodel -q").read().strip().splitlines()[-1]
    # v5 同口径: 锁频(REVIEW_P1 P1.4; MAXN 无锁 DVFS 只给 ~509MHz 非 917, 上次跑漏了这个→口径错作废)
    os.system(f"echo {os.environ.get('JETSON_SUDO_PW','nx')} | sudo -S jetson_clocks --fan")
    print("  jetson_clocks --fan 已锁(917)", flush=True)
    print(f"[AB-interleaved 25W] n={len(audios)} power={q}", flush=True)

    rfs = ResFlowSEModel.load_from_checkpoint(args.resflowse_ckpt, map_location="cpu", weights_only=False, strict=False)
    rfs.cuda().eval()
    fse = VFModel.load_from_checkpoint(args.flowse_ckpt, base_dir=os.path.dirname(os.path.dirname(args.noisy_dir.rstrip("/"))), map_location="cpu")
    try: fse.data_module.setup(stage=None)
    except Exception: pass
    for n, p in fse.dnn.named_parameters():
        if n in fse.ema_dnn: p.data = fse.ema_dnn[n].to(p.device)
    fse.cuda().eval()

    def A(Y, T):
        with torch.no_grad():
            xh = rfs.forward(Y); _ = rfs.to_audio(xh.squeeze(), T)

    def B(Y, T, N=1):
        with torch.no_grad():
            sampler = get_white_box_solver("euler", fse.ode, fse, Y, T_rev=fse.T_rev, t_eps=fse.t_eps, N=N)
            sample, _ = sampler(); _ = fse.to_audio(sample.squeeze(), T)

    # ===== 附4协议对照段: bench_jetson 同款预热(3句+连续60s) → 立即测全824 ResFlowSE(单模型块) =====
    # 复现 v5 的预热路径: 若 MAXN 上该路径也偏快(→v5同偏差可复现可定位)或一致(→偏差为25W特有, 机制未知入正文)
    print("  [协议对照] bench_jetson 式预热: 3句 + 连续60s...", flush=True)
    for y, _ in audios[:3]:
        Yr = prep_Y(rfs, y, device); A(Yr, y.size(1))
    torch.cuda.synchronize()
    t0w = time.time(); iw = 0
    while time.time() - t0w < 60:
        y, _ = audios[iw % len(audios)]; Yr = prep_Y(rfs, y, device); A(Yr, y.size(1)); iw += 1
    torch.cuda.synchronize()
    print(f"  [协议对照] 60s 预热完成({iw} iters), 立即测全824 ResFlowSE(单模型块, v5同序)...", flush=True)
    lat_proto = []
    for y, _ in audios:
        Yr = prep_Y(rfs, y, device)
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); A(Yr, y.size(1)); e.record(); torch.cuda.synchronize()
        lat_proto.append(s.elapsed_time(e))
    lat_proto = np.array(lat_proto); rtf_proto = lat_proto / 1000 / durs
    print(f"  [协议对照] v5式预热路径 全824: RTF mean {rtf_proto.mean():.5f} (v5 MAXN 块值 0.16523; 交错待测)", flush=True)

    # 交错前置预热: 两模型各 20 句交错
    for y, _ in audios[:20]:
        Yr = prep_Y(rfs, y, device); A(Yr, y.size(1))
        Yf = prep_Y(fse, y, device); B(Yf, y.size(1), 1)
    torch.cuda.synchronize()
    print("  warmup done(两模型交错20句)", flush=True)

    # ===== 逐文件交错 A/B, ABBA 轮转消位置效应 =====
    latA, latB, order_log = [], [], []
    for i, (y, dur) in enumerate(audios):
        T = y.size(1)
        Yr = prep_Y(rfs, y, device); Yf = prep_Y(fse, y, device)
        phase = (i // 1) % 4   # ABBA: i%4∈{0,1}->A first; {2,3}->B first
        seq = [("A", Yr, T, A), ("B", Yf, T, lambda Y, T: B(Y, T, 1))] if i % 4 in (0, 1) \
              else [("B", Yf, T, lambda Y, T: B(Y, T, 1)), ("A", Yr, T, A)]
        rec = {}
        for tag, Y, Tt, fn in seq:
            rec[tag] = time_fwd(lambda: fn(Y, Tt))
        latA.append(rec["A"]); latB.append(rec["B"]); order_log.append("AB" if i % 4 in (0, 1) else "BA")
        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/824] A {np.mean(latA):.1f}ms B {np.mean(latB):.1f}ms running Δ={(np.mean(latB)-np.mean(latA))/np.mean(latA)*100:+.1f}%", flush=True)
    latA = np.array(latA); latB = np.array(latB)
    rtfA = latA / 1000 / durs; rtfB = latB / 1000 / durs
    d = rtfB - rtfA
    t_, p_ = sstats.ttest_rel(rtfB, rtfA)
    diff_rel = float(d.mean() / rtfA.mean() * 100)
    ab_first = rtfA[[i for i, o in enumerate(order_log) if o == "AB"]]
    ba_first = rtfA[[i for i, o in enumerate(order_log) if o == "BA"]]
    ab_B = rtfB[[i for i, o in enumerate(order_log) if o == "AB"]]
    ba_B = rtfB[[i for i, o in enumerate(order_log) if o == "BA"]]
    print(f"\n  A(ResFlowSE) RTF mean {rtfA.mean():.5f} | B(FlowSE N=1) RTF mean {rtfB.mean():.5f}", flush=True)
    print(f"  逐文件配对差: mean {d.mean():+.5f} ({diff_rel:+.1f}%), paired t p={p_:.2e}", flush=True)
    print(f"  位置效应核(AB序 A={ab_first.mean():.5f} vs BA序 A={ba_first.mean():.5f}; AB序 B={ab_B.mean():.5f} vs BA序 B={ba_B.mean():.5f})", flush=True)
    ab = {"power": q, "n": len(audios), "resflowse_A": {"latency_ms": stats(latA), "rtf": stats(rtfA)},
          "flowse_N1_B": {"latency_ms": stats(latB), "rtf": stats(rtfB)},
          "paired_diff": {"mean": float(d.mean()), "rel_pct": diff_rel, "paired_t_p": float(p_), "ci90": [float(x) for x in sstats.t.interval(0.90, len(d)-1, loc=d.mean(), scale=d.std(ddof=1)/np.sqrt(len(d)))]},
          "position_effect": {"A_mean_AB_first": float(ab_first.mean()), "A_mean_BA_first": float(ba_first.mean()),
                              "B_mean_AB_first": float(ab_B.mean()), "B_mean_BA_first": float(ba_B.mean())},
          "verdict_note": "15%异常复核: 此前块设计两块差15%(MAXN同对1.7%); 若交错后差值回~2%→位置/会话效应; 若仍~15%→真实模型差(FlowSE N=1 确实比 ResFlowSE 单步慢)"}

    out = {"task": "MAXN 交错A/B 复测(派活3, 0.16523) + v5式预热路径协议对照(附4)", "caliber": CALIBER, **ab,
           "protocol_control_v5warmup": {"latency_ms": stats(lat_proto), "rtf": stats(rtf_proto),
                                          "note": "3句+连续60s预热(bench_jetson式)后立即测全824单模型块; 对照 v5 MAXN 块值 0.16523 与交错值: 偏快→偏差可复现可定位, 一致→偏差为25W特有机制未知"}}
    np.save(args.output.replace(".json", "_perfile.npy"),
            {"latA_ms": latA, "latB_ms": latB, "dur_s": durs,
             "rtfA": rtfA, "rtfB": rtfB, "order": order_log}, allow_pickle=True)
    out["env_before"] = ENV_BEFORE
    out["env_after"] = env_snapshot("after")
    json.dump(out, open(args.output, "w"), indent=2)
    sha = hashlib.sha256(open(args.output, "rb").read()).hexdigest()
    open(args.output + ".sha256", "w").write(f"{sha}  {args.output}\n")
    print(f"\n✓ saved {args.output} sha={sha[:16]}", flush=True)


if __name__ == "__main__":
    main()
