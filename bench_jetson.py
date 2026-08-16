"""bench_jetson.py v5 — Jetson Orin NX 单档基准(PLAN_FINAL v3 + REVIEW P1 必改1 + mentor2 实验B负载改)。

口径(caliber, 与 A100 bench_t2a.py 一致):
  latency = cuda.Event( network_forward(model.forward / euler-solver NFE) + iSTFT(to_audio) );
  输入复数谱 Y 预计算(STFT+spec_fwd+pad_spec 不计时); batch=1; sf.read(48k)->resample(16k)。
  dur 用 sf.info().frames/samplerate 实际 sr 算(不硬编码 16000 —— 曾致 RTF 错 3 倍)。

P0 能耗修复(必改1/2): 逐句累加 E=Σ(P_i×lat_i)/Σdur_i; 跨口径独立恒等式。
P1 跑序(PLAN_FINAL v3 + mentor2):
  idle冷 → ResFlowSE单步全824(cold: headline RTF + 单步能耗) → to_audio全824(冷)
  → FlowSE加载 → 自洽闸门(<5%) → 【实验A#1 冷扫 子集NFE1..6 → N_max_cold】 → 子集校准 → N=2线性闸门
  → 【实验B: FlowSE@N=N_max_cold(p95) 满824循环≥drift_min(临界配置连续跑, mentor2改)】 → idle热
  → 【实验A#2 热扫】 → 恒等式 → json+sha256 → scp回本机 → 入ledger
  理由(mentor2): 单步RTF仅0.176(MAXN)离RTF=1有5.7×余量, 漂移推不过线只给幅度; 跑临界配置才答"你真会选的配置连跑1h还成立吗"。
闸门FAIL(自洽/N=2线性/校准)→ 立即停, 跳过热扫, 报导师不硬往下。
Jetson 专属件(tegrastats/nvpmodel/sudo)优雅降级, 可在 A100 smoke 验逻辑。
"""
import argparse, json, os, sys, time, subprocess, re, threading, hashlib
import torch, numpy as np, soundfile as sf, torchaudio
import torch.serialization
os.environ['PATH'] = os.path.expanduser('~/.local/bin') + ':/usr/local/cuda/bin:' + os.environ['PATH']
from glob import glob

SR = 16000
SUDO_PW = os.environ.get("JETSON_SUDO_PW", "nx")
CALIBER = ("latency = cuda.Event(network_forward(model.forward or euler-solver NFE) + iSTFT(to_audio)); "
           "input complex spectrogram Y precomputed (STFT+spec_fwd+pad_spec outside timing); batch=1; "
           "audio loaded sf.read(48k)->torchaudio.resample(16k) same path as A100")


def _is_jetson():
    try:
        return "nvidia" in open("/proc/device-tree/compatible", "rb").read().decode("utf-8", "ignore")
    except Exception:
        return False


def _stub_onnxruntime_if_needed():
    """15W/10W 档 CPU 只上 4 核 → onnxruntime import 即崩(cpuid assert); 推理不需要它(仅 torchmetrics 可选依赖)。 在 4 核时用桩断链, 8 核零影响。桩已带 __spec__ 过 importlib 探测。"""
    import os as _os
    if _os.cpu_count() and _os.cpu_count() < 8:
        import sys as _sys, types as _types, importlib as _il
        if "onnxruntime" not in _sys.modules:
            spec = _il.machinery.ModuleSpec("onnxruntime", None)
            stub = _types.ModuleType("onnxruntime"); stub.__spec__ = spec
            stub.__version__ = "0.0.0-stub(4-core-mode: real ort crashes at cpuid assert)"
            def _b(*a, **k): raise ImportError("onnxruntime stubbed (4-core mode)")
            stub.InferenceSession = _b
            stub.get_available_providers = lambda: ["CPUExecutionProvider"]
            _sys.modules["onnxruntime"] = stub
            print("  [caliber] CPU核数<8(15W/10W 档) → onnxruntime 桩断链(真实 ort 在 4 核下 import 崩, 推理不需要)")


_stub_onnxruntime_if_needed()


ON_JETSON = _is_jetson()


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



def sh(cmd, sudo=False, timeout=120):
    if sudo:
        cmd = f"echo {SUDO_PW} | sudo -S -k bash -c '{cmd}'"
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return r.stdout + r.stderr
    except Exception as e:
        return f"<sh error: {e}>"


def set_power(mode_id):
    return sh("nvpmodel -q").strip() if ON_JETSON else f"<not jetson; requested mode_id={mode_id}>"


def set_clocks(on):
    if not ON_JETSON:
        return "<not jetson; clocks n/a>"
    if on:
        sh("jetson_clocks --fan", sudo=True)
    return sh("jetson_clocks --show", sudo=True)[:200]


def parse_vdd(line):
    m = re.search(r'VDD_IN (\d+)mW', line); return int(m.group(1)) if m else None


def parse_thermal(line):
    g = re.search(r'gpu@([\d.]+)C', line); f = re.search(r'GR3D_FREQ \d+%@\[(\d+)\]', line)
    return (float(g.group(1)) if g else None, int(f.group(1)) if f else None)


class TegrastatsRecorder:
    def __init__(self, interval_ms=100):
        self.interval_ms = interval_ms; self.samples = []
        self._proc = None; self._thread = None; self._running = False; self.available = ON_JETSON

    def start(self):
        self.samples = []; self._running = True
        if not self.available:
            return
        try:
            self._proc = subprocess.Popen(f"echo {SUDO_PW} | sudo -S tegrastats --interval {self.interval_ms}",
                                          shell=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            self._thread = threading.Thread(target=self._read, daemon=True); self._thread.start()
        except Exception as e:
            print(f"  [warn] tegrastats 启动失败: {e}; 能耗降级为 None"); self.available = False

    def _read(self):
        while self._running:
            line = self._proc.stdout.readline()
            if not line:
                break
            ts = time.time(); v = parse_vdd(line); th = parse_thermal(line)
            self.samples.append((ts, v, th[0], th[1]))

    def stop(self):
        self._running = False
        if self._proc:
            try:
                self._proc.terminate()
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=2)

    def power_window(self, t0, t1):
        if not self.available:
            return None
        ws = [s[1] for s in self.samples if s[1] is not None and t0 <= s[0] <= t1]
        return float(np.mean(ws)) if ws else None

    def idle_stats(self):
        if not self.samples:
            return None
        v = np.array([s[1] for s in self.samples if s[1] is not None], float)
        th = np.array([s[2] for s in self.samples if s[2] is not None], float)
        fq = np.array([s[3] for s in self.samples if s[3] is not None], float)
        return {"n": len(self.samples), "vdd_mW": stats(v) if len(v) else None,
                "temp_C": ({"mean": float(th.mean()), "max": float(th.max())} if len(th) else None),
                "freq_MHz": ({"min": int(fq.min()), "max": int(fq.max())} if len(fq) else None),
                "interval_ms": self.interval_ms}


def load_audio(fp, sr=SR):
    info = sf.info(fp); dur = info.frames / info.samplerate
    y, sr0 = sf.read(fp); y = torch.from_numpy(y).float()
    if y.dim() == 1:
        y = y.unsqueeze(0)
    if sr0 != sr:
        y = torchaudio.functional.resample(y, sr0, sr)
    return y, dur


def prep_Y(model, y_wav, device):
    from flowmse.util.other import pad_spec
    norm = y_wav.abs().max(); yn = y_wav / norm
    Y = torch.unsqueeze(model._forward_transform(model._stft(yn.to(device))), 0)
    return pad_spec(Y), norm


def stats(a):
    a = np.asarray(a, float)
    return {"n": int(len(a)), "mean": float(a.mean()), "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)), "p99": float(np.percentile(a, 99)), "std": float(a.std())}


def energy_per_utt(per_utt_power_mW, per_utt_latency_s, per_utt_dur_s, idle_mW):
    P = np.asarray(per_utt_power_mW, float); L = np.asarray(per_utt_latency_s, float)
    D = np.asarray(per_utt_dur_s, float); audio_total = float(D.sum())
    if audio_total <= 0 or len(P) == 0:
        return None
    Pc = np.where(np.isfinite(P), P, np.nan)
    E_total = float(np.nansum(Pc * L) / audio_total)
    E_incr = float(np.nansum((Pc - idle_mW) * L) / audio_total)
    per_file = ((Pc - idle_mW) * L) / np.where(D > 0, D, np.nan)
    pf = per_file[np.isfinite(per_file)]
    return {"E_total_mJ_per_s_audio": E_total, "E_incr_mJ_per_s_audio": E_incr,
            "audio_total_s": audio_total, "n": int(len(P)),
            "per_file_E_incr_mJ_per_s": stats(pf) if pf.size else None}


def identity_checks(per_utt_latency_s, per_utt_dur_s, wall_s_total, per_utt_power_mW):
    L = np.asarray(per_utt_latency_s, float); D = np.asarray(per_utt_dur_s, float)
    P = np.asarray(per_utt_power_mW, float)
    sum_lat = float(np.nansum(L)); audio_total = float(D.sum()); wall = float(wall_s_total)
    out = {"sum_latency_s": sum_lat, "wall_s_total": wall}
    out["lat_over_wall"] = (sum_lat / wall) if wall > 0 else None
    out["scheduling_gap_s"] = wall - sum_lat
    Pc = np.where(np.isfinite(P), P, np.nan)
    if audio_total > 0:
        E_pu = float(np.nansum(Pc * L) / audio_total)
        E_g = float((np.nanmean(Pc) * wall) / audio_total)
        out.update({"E_per_utt_path": E_pu, "E_global_path": E_g, "E_path_ratio": (E_pu / E_g) if E_g else None})
    if audio_total > 0 and (D > 0).all():
        rtf_pf = L / D; mom = sum_lat / audio_total
        out.update({"rtf_per_file_mean": float(np.nanmean(rtf_pf)), "rtf_mean_over_mean": float(mom),
                    "rtf_jensen_ratio": (float(np.nanmean(rtf_pf)) / mom) if mom else None})
    return out


def drift_windows(rtf_records, teg_samples, bin_s=300):
    """5min 窗漂移(REVIEW_P1 P1.6/M22.1): 剔 n_rtf<0.5×中位 的不完整尾窗; 每窗附 rtf_se; 首末差±合并SE; 不解读为热降频(零结果=观察到非证明)。"""
    if not rtf_records:
        return None
    t0 = rtf_records[0][0]
    rbin = {}
    for ts, lat, dur in rtf_records:
        rbin.setdefault(int((ts - t0) // bin_s), []).append((lat, dur))
    tbin = {}
    for ts, vdd, temp, freq in (teg_samples or []):
        if temp is not None or freq is not None:
            tbin.setdefault(int((ts - t0) // bin_s), []).append((temp, freq))
    out = []
    for b in sorted(set(rbin) | set(tbin)):
        rb = rbin.get(b, [])
        tb = np.array(tbin.get(b, []), float) if tbin.get(b) else np.array([]).reshape(0, 2)
        rtf = np.array([l / 1000 / d for l, d in rb]) if rb else np.array([])
        out.append({"window": b, "n_rtf": len(rb),
                    "rtf_mean": float(np.nanmean(rtf)) if rtf.size else None,
                    "rtf_se": (float(np.std(rtf, ddof=1) / np.sqrt(len(rtf))) if rtf.size > 1 else None),
                    "temp_C_mean": float(np.nanmean(tb[:, 0])) if tb.size else None,
                    "freq_MHz_mean": (float(np.nanmean(tb[:, 1][tb[:, 1] > 0])) if tb.size and (tb[:, 1] > 0).any() else None)})
    nrtfs = [w["n_rtf"] for w in out if w["n_rtf"] > 0]
    thr = 0.5 * float(np.median(nrtfs)) if nrtfs else 0.0
    for w in out:
        w["valid"] = bool(w["n_rtf"] >= thr) if thr else True
    vw = [w for w in out if w["valid"] and w["rtf_mean"] is not None]
    if len(vw) >= 2:
        f, l = vw[0], vw[-1]
        se_f = f.get("rtf_se") or 0.0; se_l = l.get("rtf_se") or 0.0
        comb_se = (se_f ** 2 + se_l ** 2) ** 0.5 if (se_f or se_l) else None
        diff = l["rtf_mean"] - f["rtf_mean"]; rel = (diff / f["rtf_mean"]) if f["rtf_mean"] else None
        vm = [w["rtf_mean"] for w in vw]
        out.append({"summary": True, "n_valid_windows": len(vw), "n_excluded_incomplete": len(out) - len(vw),
                    "n_rtf_threshold": thr, "first_window_rtf": f["rtf_mean"], "last_window_rtf": l["rtf_mean"],
                    "first_last_rtf_diff": diff, "first_last_rel_pct": (rel * 100) if rel is not None else None,
                    "first_last_diff_pm_SE": (f"{diff:.5f} ± {comb_se:.5f}" if comb_se is not None else None),
                    "window_rtf_mean_std_rel_pct": (float(np.std(vm) / np.mean(vm)) * 100) if vm else None,
                    "note": "零结果: 未观察到热致退化(锁频条件下); 不解读为热降频(措辞:观察到非证明); 默认DVFS未测入§8"})
    return out


def select_subset(audios, n):
    if n >= len(audios):
        return audios
    order = np.argsort([d for _, d in audios])
    idx = np.linspace(0, len(order) - 1, n, dtype=int)
    return [audios[order[i]] for i in idx]


def run_nfe_sweep(fse, fse_fn, prep_Y, device, subset, sub_durs, nfe_to_run, tag=""):
    sweep = {}
    for N in nfe_to_run:
        for y, _ in subset[:3]:
            Y, _ = prep_Y(fse, y, device); fse_fn(Y, y.size(1), N)()
        torch.cuda.synchronize()
        lat = []
        for y, _ in subset:
            T = y.size(1); Y, _ = prep_Y(fse, y, device)
            lat.append(time_fwd_cuda(fse_fn(Y, T, N))[0])
        lat = np.array(lat); rtf = lat / 1000 / sub_durs
        sweep[f"N={N}"] = {"latency_ms": stats(lat), "rtf": stats(rtf),
                           "rtf_mean_over_mean": float(lat.sum() / 1000 / sub_durs.sum())}
        print(f"  [{tag}] FlowSE N={N}: lat {lat.mean():.2f}ms RTF mean {rtf.mean():.5f} p95 {np.percentile(rtf,95):.5f}")
    return sweep


def nmax_from_sweep(sweep, stat_key):
    ns = [int(k.split("=")[1]) for k, v in sweep.items() if (v["rtf"][stat_key] or 1) < 1.0]
    return max(ns) if ns else 0


def run_load_loop(step_fn, audios, interval_ms, drift_min, tag=""):
    """满 audios 循环 step_fn(y,dur)->(lat_ms,t0,t1) 直至 drift_min 分钟(0=1pass); tegrastats 后台。model-agnostic。"""
    rec = TegrastatsRecorder(interval_ms); rec.start()
    wall0 = time.time(); lat, wall, dur, pw, drift_records = [], [], [], [], []
    pass_n = 0
    while True:
        for y, d in audios:
            lat_ms, t0, t1 = step_fn(y, d)
            lat.append(lat_ms); wall.append(t1 - t0); dur.append(d); pw.append(rec.power_window(t0, t1))
            drift_records.append((t0, lat_ms, d))
        pass_n += 1
        if drift_min <= 0 or time.time() - wall0 >= drift_min * 60 or (pass_n >= 1 and not ON_JETSON):
            break
    rec.stop()
    print(f"  [{tag}] {pass_n} pass, {len(lat)} 句, wall={time.time()-wall0:.1f}s")
    return np.array(lat), np.array(wall), np.array(dur), pw, drift_records, time.time() - wall0, list(rec.samples), pass_n


def time_fwd_cuda(fn):
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    t0 = time.time()
    s.record(); fn(); e.record(); torch.cuda.synchronize()
    t1 = time.time()
    return s.elapsed_time(e), (t0, t1)


def sha256_file(path, head=16):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()[:head]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode_id", type=int, required=True, help="0=MAXN 1=10W 2=15W 3=25W")
    ap.add_argument("--clocks", choices=["on", "off"], default="on")
    ap.add_argument("--noisy_dir", default=os.path.expanduser("~/sflowse_pkg/data/test/noisy"))
    ap.add_argument("--parity_n", type=int, default=20)
    ap.add_argument("--n_real", type=int, default=824)
    ap.add_argument("--nfe_list", default="1,2,3,4,5,6")
    ap.add_argument("--subset_n", type=int, default=100)
    ap.add_argument("--drift_min", type=int, default=60, help="实验B FlowSE@N_max_cold 漂移循环目标分钟(0=1pass)")
    ap.add_argument("--calib_full824", action="store_true")
    ap.add_argument("--resflowse_ckpt", default=os.path.expanduser("~/sflowse.ckpt"))
    ap.add_argument("--flowse_ckpt", default=os.path.expanduser("~/VB_DMD_FLOWSE_ICASSP_2025.ckpt"))
    ap.add_argument("--tegrastats_interval_ms", type=int, default=100)
    ap.add_argument("--idle_cold_s", type=int, default=120)
    ap.add_argument("--idle_hot_s", type=int, default=60)
    ap.add_argument("--self_consistency_max_pct", type=float, default=5.0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    device = "cuda"
    from flowmse.util.other import pad_spec
    from flowmse.resflowse_model import ResFlowSEModel
    nfe_list = sorted(set(int(x) for x in args.nfe_list.split(",")))
    import importlib as _il
    Uop = _il.import_module("flowmse.backbones.ncsnpp_utils.op.upfirdn2d")

    data_dir = os.path.dirname(os.path.dirname(args.noisy_dir.rstrip("/")))
    noisy = sorted(glob(os.path.join(args.noisy_dir, "*.wav")))
    n_real = min(args.n_real, len(noisy)); noisy = noisy[:n_real]
    audios = [load_audio(nf) for nf in noisy]
    durs = np.array([d for _, d in audios])
    print(f"[bench_jetson v5 mode={args.mode_id}] real={n_real} mean_dur={durs.mean():.4f}s on_jetson={ON_JETSON}")

    ENV_BEFORE = env_snapshot("before")
    power_q = set_power(args.mode_id)
    clocks_show = set_clocks(args.clocks == "on")
    print("  power:", (power_q.splitlines()[-1] if power_q else "?"))

    # idle 冷态
    print(f"  idle 冷态 {args.idle_cold_s}s ...")
    rec_cold = TegrastatsRecorder(args.tegrastats_interval_ms); rec_cold.start()
    time.sleep(args.idle_cold_s); rec_cold.stop()
    idle_cold = rec_cold.idle_stats()
    idle_cold_mW = idle_cold["vdd_mW"]["mean"] if (idle_cold and idle_cold.get("vdd_mW")) else None

    # ResFlowSE 加载 + warmup
    rfs = ResFlowSEModel.load_from_checkpoint(args.resflowse_ckpt, map_location="cpu", weights_only=False, strict=False)
    rfs.cuda().eval()

    def rfs_fn(Y, T):
        def _f():
            with torch.no_grad():
                xh = rfs.forward(Y); _ = rfs.to_audio(xh.squeeze(), T)
        return _f

    for y, _ in audios[:3]:
        Y, _ = prep_Y(rfs, y, device); rfs_fn(Y, y.size(1))()
    torch.cuda.synchronize(); t0 = time.time(); i = 0
    while time.time() - t0 < 60:
        y, _ = audios[i % len(audios)]; Y, _ = prep_Y(rfs, y, device); rfs_fn(Y, y.size(1))(); i += 1
    print(f"  ResFlowSE warmup 60s done ({i} iters)")

    # ResFlowSE 单步 全824(cold: headline RTF + 单步能耗)
    def rfs_step(y, dur):
        T = y.size(1); Y, _ = prep_Y(rfs, y, device); lat_ms, (ta, tb) = time_fwd_cuda(rfs_fn(Y, T)); return lat_ms, ta, tb
    rfs_lat, rfs_wall, rfs_dur_b, rfs_pow, _, rfs_wt, _, rfs_pn = run_load_loop(
        rfs_step, audios, args.tegrastats_interval_ms, 0, "ResFlowSE单步全824")
    rfs_rtf = rfs_lat / 1000 / rfs_dur_b
    rfs_result = {"latency_ms": stats(rfs_lat), "rtf": stats(rfs_rtf),
                  "rtf_mean_over_mean": float(rfs_lat.sum() / 1000 / rfs_dur_b.sum()),
                  "n_passes": rfs_pn, "wall_s_total": rfs_wt}
    single_step_E = energy_per_utt(rfs_pow, rfs_wall, rfs_dur_b, idle_cold_mW) if idle_cold_mW else None
    print(f"  ResFlowSE 单步: RTF mean {rfs_rtf.mean():.5f} p95 {np.percentile(rfs_rtf,95):.5f}")

    # to_audio 全824(冷; 与差分估计 0.642ms 同语料对照)
    toa_lat = []
    for y, _ in audios:
        T = y.size(1); Y, _ = prep_Y(rfs, y, device)
        with torch.no_grad():
            xh = rfs.forward(Y)
        spec, Tt = xh.squeeze(), T
        toa_lat.append(time_fwd_cuda(lambda: rfs.to_audio(spec, Tt))[0])
    toa_result = {"to_audio_latency_ms": stats(np.array(toa_lat)),
                  "note": "全824 直接 cuda.Event iSTFT; 对照 ledger 差分估计 0.642ms; 对照前不判差分被证伪"}

    # FlowSE 加载 + warmup
    from flowmse.data_module import SpecsDataModule
    from flowmse.model import VFModel
    from flowmse.sampling import get_white_box_solver
    try:
        torch.serialization.add_safe_globals([SpecsDataModule])
    except AttributeError:
        pass
    fse = VFModel.load_from_checkpoint(args.flowse_ckpt, base_dir=data_dir, map_location="cpu")
    try:
        fse.data_module.setup(stage=None)
    except Exception:
        pass
    for name, p in fse.dnn.named_parameters():
        if name in fse.ema_dnn:
            p.data = fse.ema_dnn[name].to(p.device)
    fse.cuda().eval()
    T_rev, t_eps = fse.T_rev, fse.t_eps

    def fse_fn(Y, T, N):
        def _f():
            with torch.no_grad():
                sampler = get_white_box_solver("euler", fse.ode, fse, Y, T_rev=T_rev, t_eps=t_eps, N=N)
                sample, _ = sampler(); _ = fse.to_audio(sample.squeeze(), T)
        return _f

    for y, _ in audios[:3]:
        Y, _ = prep_Y(fse, y, device); fse_fn(Y, y.size(1), 1)()
    torch.cuda.synchronize()
    print("  FlowSE 加载+warmup done")

    # 自洽闸门(冷): FlowSE N=1 vs ResFlowSE 单步
    gate_n = min(20, len(audios))
    rfs_lat_gate = []; fN1_lat = []
    for y, _ in audios[:gate_n]:
        T = y.size(1); Yr, _ = prep_Y(rfs, y, device); Yf, _ = prep_Y(fse, y, device)
        rfs_lat_gate.append(time_fwd_cuda(rfs_fn(Yr, T))[0]); fN1_lat.append(time_fwd_cuda(fse_fn(Yf, T, 1))[0])
    rfs_g = float(np.mean(rfs_lat_gate)); fN1_m = float(np.mean(fN1_lat))
    self_con = {"resflowse_single_ms": rfs_g, "flowse_N1_ms": fN1_m,
                "rel_diff_pct": abs(rfs_g - fN1_m) / max(rfs_g, fN1_m) * 100,
                "gate_lt5pct": bool(abs(rfs_g - fN1_m) / max(rfs_g, fN1_m) * 100 < args.self_consistency_max_pct)}
    print(f"  自洽闸门: {rfs_g:.2f} vs {fN1_m:.2f}ms → {self_con['rel_diff_pct']:.2f}% [{'PASS' if self_con['gate_lt5pct'] else 'FAIL-停'}]")
    nfe_to_run = nfe_list if self_con["gate_lt5pct"] else [1]

    # 实验 A #1 冷扫 → N_max_cold
    subset = select_subset(audios, args.subset_n)
    sub_durs = np.array([d for _, d in subset])
    sweep_cold = run_nfe_sweep(fse, fse_fn, prep_Y, device, subset, sub_durs, nfe_to_run, tag="冷扫")
    linearity = None
    if "N=1" in sweep_cold and "N=2" in sweep_cold:
        n2 = sweep_cold["N=2"]["latency_ms"]["mean"]; n1 = sweep_cold["N=1"]["latency_ms"]["mean"]; per_step = n2 - n1
        linearity = {"per_step_N2minusN1_ms": per_step, "n1_ms": n1, "n2_ms": n2, "n2_over_2x_n1": n2 / (2 * n1),
                     "linear_ok": bool(abs(per_step - n1) / max(n1, 1e-9) < 0.15), "note": "N=2 偏离2×per_step→停报导师"}
        print(f"  N=2 线性: per_step={per_step:.2f} vs N1={n1:.2f}ms → {'PASS' if linearity['linear_ok'] else 'FAIL-停'}")
    n_max_cold = {"N_max_mean": nmax_from_sweep(sweep_cold, "mean"), "N_max_p95": nmax_from_sweep(sweep_cold, "p95")}
    print(f"  N_max_cold: {n_max_cold}")

    # 子集校准
    calib = None
    if args.calib_full824 and "N=1" in sweep_cold:
        print("  校准: FlowSE N=1 全824 ...")
        lat_full = []
        for y, _ in audios:
            T = y.size(1); Y, _ = prep_Y(fse, y, device); lat_full.append(time_fwd_cuda(fse_fn(Y, T, 1))[0])
        lat_full = np.array(lat_full); rtf_full = lat_full / 1000 / durs
        full_mean = float(rtf_full.mean()); full_se = float(rtf_full.std() / np.sqrt(len(rtf_full)))
        sub_mean = sweep_cold["N=1"]["rtf"]["mean"]; delta = abs(full_mean - sub_mean); tol = max(2 * full_se, 0.01 * full_mean)
        calib = {"full824_n1_rtf_mean": full_mean, "full824_n1_rtf_se": full_se, "subset_n1_rtf_mean": sub_mean,
                 "abs_delta": delta, "tolerance": tol, "calibrated_ok": bool(delta <= tol)}
        print(f"  校准: full824={full_mean:.5f} vs subset={sub_mean:.5f} Δ={delta:.2e} tol={tol:.2e} → {'PASS' if calib['calibrated_ok'] else 'FAIL'}")

    # 实验 B: FlowSE @ N=N_max_cold(p95) 满824循环≥drift_min(mentor2 改: 临界配置连续跑)
    N_bench = n_max_cold["N_max_p95"]
    bench_cfg = {"N": N_bench, "stat": "p95", "source": "n_max_cold from 冷扫",
                 "note": "实验B负载=N_max_cold(p95); 临界配置连续跑→答'你真会选的配置连跑1h还成立吗'"}
    b_lat = b_wall = b_dur = np.array([]); b_pow = []; drift_records = []; wall_b_total = 0.0; teg_samples_b = []; pass_n = 0
    bench_result = None
    if N_bench and N_bench > 0:
        def fse_step(y, dur, _N=N_bench):
            T = y.size(1); Y, _ = prep_Y(fse, y, device); lat_ms, (ta, tb) = time_fwd_cuda(fse_fn(Y, T, _N)); return lat_ms, ta, tb
        b_lat, b_wall, b_dur, b_pow, drift_records, wall_b_total, teg_samples_b, pass_n = run_load_loop(
            fse_step, audios, args.tegrastats_interval_ms, args.drift_min, f"实验B FlowSE N={N_bench}")
        bench_rtf = b_lat / 1000 / b_dur
        bench_result = {"config": bench_cfg, "latency_ms": stats(b_lat), "rtf": stats(bench_rtf),
                        "rtf_mean_over_mean": float(b_lat.sum() / 1000 / b_dur.sum()), "n_passes": pass_n, "wall_s_total": wall_b_total}
        # 存逐句 RTF(供 P(RTF≥1) 实测比例; REVIEW_P1 P1.1) + 算 P(RTF≥1)
        np.save(args.output.replace('.json', '_expB_perfile.npy'),
                {'lat_ms': b_lat, 'dur_s': b_dur, 'rtf': bench_rtf}, allow_pickle=True)
        bench_result['P_RTF_ge_1'] = float(np.mean(bench_rtf >= 1.0))
        bench_result['P_RTF_ge_1_note'] = '实测逐句 RTF≥1 的比例(n=824×passes); 比 p95 更直接的部署量'
    else:
        bench_result = {"config": bench_cfg, "skipped": "N_max_cold=0/None 无可跑临界配置"}
        print("  ⚠️ 实验B跳过: N_max_cold=0/None")

    # 能耗(单步 from rfs pass; N_max_cold from 实验 B; 冷/热 idle 区间)
    energy = {"idle_cold": idle_cold, "single_step_E_using_idle_cold": single_step_E}
    if any(p is not None for p in b_pow) and idle_cold_mW:
        energy["nmax_cold_E_using_idle_cold"] = energy_per_utt(b_pow, b_wall, b_dur, idle_cold_mW)

    # idle 热态
    if ON_JETSON and args.drift_min > 0:
        print(f"  idle 热态 {args.idle_hot_s}s ...")
        rec_hot = TegrastatsRecorder(args.tegrastats_interval_ms); rec_hot.start()
        time.sleep(args.idle_hot_s); rec_hot.stop()
        idle_hot = rec_hot.idle_stats(); energy["idle_hot"] = idle_hot
        idle_hot_mW = idle_hot["vdd_mW"]["mean"] if (idle_hot and idle_hot.get("vdd_mW")) else None
        if idle_hot_mW:
            if single_step_E is not None:
                energy["single_step_E_using_idle_hot"] = energy_per_utt(rfs_pow, rfs_wall, rfs_dur_b, idle_hot_mW)
            if any(p is not None for p in b_pow):
                energy["nmax_cold_E_using_idle_hot"] = energy_per_utt(b_pow, b_wall, b_dur, idle_hot_mW)
            energy["E_incr_interval_note"] = "增量能耗给区间[冷idle,热idle]; 热>冷则冷idle口径高估增量"
    energy["note"] = ("VDD_IN 整板; total=不扣idle incremental=total−idle; per-utterance streaming workload NOT sustained saturation; "
                      "单步能耗(ResFlowSE)与 N_max_cold 能耗(FlowSE@临界配置)分开报")

    # 实验 A #2 热扫(闸门全 PASS 才跑)
    hot_ok = bool(self_con["gate_lt5pct"] and (linearity is None or linearity["linear_ok"]) and (calib is None or calib["calibrated_ok"]))
    if hot_ok:
        sweep_hot = run_nfe_sweep(fse, fse_fn, prep_Y, device, subset, sub_durs, nfe_to_run, tag="热扫")
    else:
        sweep_hot = {}; print("  ⚠️ 热扫跳过: 闸门FAIL, 标 STOP 报导师")
    n_max_hot = ({"N_max_mean": nmax_from_sweep(sweep_hot, "mean"), "N_max_p95": nmax_from_sweep(sweep_hot, "p95")}
                 if sweep_hot else {"N_max_mean": None, "N_max_p95": None, "skipped": "闸门FAIL"})

    # 恒等式(用 实验 B FlowSE@N_max_cold 数据)
    idc = identity_checks(b_wall, b_dur, wall_b_total, b_pow) if len(b_wall) else None
    if idc and len(b_lat):
        idc["sum_cuda_latency_s"] = float(b_lat.sum() / 1000)
        idc["cuda_over_wall"] = float(b_lat.sum() / 1000 / wall_b_total) if wall_b_total else None

    # parity(ResFlowSE)
    parity = None
    if args.parity_n > 0:
        parity_out = {}
        for nf in noisy[:args.parity_n]:
            y, _ = load_audio(nf); T = y.size(1); norm = y.abs().max(); yn = y / norm
            Y = torch.unsqueeze(rfs._forward_transform(rfs._stft(yn.to(device))), 0); Y = pad_spec(Y)
            with torch.no_grad():
                xh = rfs.forward(Y)
            xhat = (rfs.to_audio(xh.squeeze(), T) * norm).squeeze().cpu().numpy().astype(np.float32)
            parity_out[os.path.basename(nf)] = xhat
        np.save(args.output.replace('.json', '_parity.npy'), parity_out)
        parity = {"n": args.parity_n, "output_npy": args.output.replace('.json', '_parity.npy'), "vs_a100": "pending"}

    out = {
        "caliber": CALIBER,
        "platform": {"device": torch.cuda.get_device_name(0), "torch": torch.__version__, "on_jetson": ON_JETSON, "precision": "fp32", "online_cpu_cores": os.cpu_count(), "cpu_cores_note": "Orin 功耗档定义在线核数: MAXN/25W=8, 15W=4, 10W=? — 档位属性与频率并列记录; 15W 固定开销(~320ms CPU侧)可能显著变大, RTF 非 GPU 频率缩放",
                     "tf32": {"matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32), "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32)},
                     "tegrastats_interval_ms": args.tegrastats_interval_ms,
                     "ncsnpp_op_path": "pure_pytorch" if Uop.upfirdn2d_op is None else "cuda",
                     "NCSNPP_env": {k: os.environ.get(k, "0") for k in ["NCSNPP_PURE_PYTORCH", "NCSNPP_PURE_UPFIRDN", "NCSNPP_SKIP_CUDA_LOAD"]}},
        "power_mode": {"id": args.mode_id, "nvpmodel_q": power_q, "jetson_clocks": args.clocks, "clocks_show": clocks_show},
        "audio": {"sr_hz": 48000, "mean_dur_s": float(durs.mean()), "n": n_real},
        "resflowse_1nfe": rfs_result,
        "to_audio_direct": toa_result,
        "flowse_self_consistency": self_con,
        "flowse_nfe_sweep_cold": {"n_subset": len(subset), "nfe": nfe_to_run, **sweep_cold},
        "flowse_nfe_sweep_hot": ({"n_subset": len(subset), "nfe": nfe_to_run, **sweep_hot} if sweep_hot else {"skipped": "闸门FAIL"}),
        "subset_calibration": calib,
        "n2_linearity_check": linearity,
        "n_max_cold": n_max_cold,
        "n_max_hot": n_max_hot,
        "experiment_B_load": bench_result,   # FlowSE@N_max_cold(p95) 临界配置连续跑
        "drift": {"windows_5min": drift_windows(drift_records, teg_samples_b), "n_records": len(drift_records), "drift_min_target": args.drift_min} if drift_records else None,
        "energy": energy,
        "identity_checks": idc,
        "parity_check": parity,
        "args": vars(args),
        "note": ("v5: 逐句能耗(必改1)+恒等式(必改2)+实验A冷/热双扫(必改1)+子集校准+实验B负载=N_max_cold(p95)(mentor2)"
                 "+to_audio全824+冷热idle+single-step与N_max_cold双能耗; FlowSE=VB_DMD ResFlowSE=sflowse"),
    }
    out["env_before"] = ENV_BEFORE
    out["env_after"] = env_snapshot("after")
    json.dump(out, open(args.output, "w"), indent=2)   # 定稿, 无自指 sha(REVIEW_T7q_R6 纪律1: 自指 sha 结构不可复核)
    # 旁挂 sha(标准 sha256sum 格式, 可 sha256sum -c 验证)
    subprocess.run(f"sha256sum {args.output} > {args.output}.sha256", shell=True)
    with open(args.output + ".sha256") as f:
        sidecar_sha = f.read().split()[0]
    print(f"\n✓ saved {args.output}  sidecar_sha256={sidecar_sha[:16]}... (full in {args.output}.sha256; 可 sha256sum -c 验证)")
    print(json.dumps({"rfs_rtf_mean": rfs_result["rtf"]["mean"], "rfs_rtf_p95": rfs_result["rtf"]["p95"],
                      "to_audio_ms": toa_result["to_audio_latency_ms"]["mean"], "self_con_pct": self_con["rel_diff_pct"],
                      "n_max_cold": n_max_cold, "n_max_hot": n_max_hot, "expB_N_bench": N_bench,
                      "expB_rtf_mean": (bench_result.get("rtf", {}) or {}).get("mean") if bench_result else None,
                      "calib_ok": (calib or {}).get("calibrated_ok")}, indent=2))


if __name__ == "__main__":
    main()
