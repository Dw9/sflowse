"""nfe_energy_n1to5.py — R2 补测(2026-08-20 用户批): FlowSE N=1..5 全824 per-N E_incr @ MAXN。
镜像 maxn_n6_confirm.py / bench_jetson 骨架(铁律15 入口一致); 本会话自测 idle 双基线; 每 N 增量落盘。"""
import os, sys, json, hashlib, time, glob
import numpy as np, torch
sys.path.insert(0, os.path.expanduser("~/sflowse_pkg"))
os.environ['PATH'] = os.path.expanduser('~/.local/bin') + ':/usr/local/cuda/bin:' + os.environ['PATH']
from bench_jetson import env_snapshot, time_fwd_cuda, TegrastatsRecorder, load_audio, prep_Y, energy_per_utt

OUT = os.path.expanduser("~/nfe_energy_n1to5.json")
ENV_BEFORE = env_snapshot("before")
os.system(f"echo {os.environ.get('JETSON_SUDO_PW','nx')} | sudo -S nvpmodel -m 0")
time.sleep(2)
os.system(f"echo {os.environ.get('JETSON_SUDO_PW','nx')} | sudo -S jetson_clocks --fan")
nvp_q = os.popen("nvpmodel -q 2>/dev/null | tail -1").read().strip()
print(f"power: {nvp_q} | cores: {os.cpu_count()}", flush=True)

device = "cuda"
rec_cold = TegrastatsRecorder(100); rec_cold.start(); time.sleep(120); rec_cold.stop()
idle_cold = rec_cold.idle_stats()
idle_cold_mW = idle_cold["vdd_mW"]["mean"] if (idle_cold and idle_cold.get("vdd_mW")) else None
print(f"idle_cold {idle_cold_mW:.1f} mW", flush=True)

from flowmse.data_module import SpecsDataModule
from flowmse.model import VFModel
from flowmse.sampling import get_white_box_solver
try: torch.serialization.add_safe_globals([SpecsDataModule])
except AttributeError: pass
fse = VFModel.load_from_checkpoint(os.path.expanduser("~/VB_DMD_FLOWSE_ICASSP_2025.ckpt"),
                                   base_dir=os.path.expanduser("~/sflowse_pkg/data"), map_location="cpu")
try: fse.data_module.setup(stage=None)
except Exception: pass
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

noisy = sorted(glob.glob(os.path.expanduser("~/sflowse_pkg/data/test/noisy/*.wav")))
loaded = [load_audio(nf) for nf in noisy]
audios = [y for y, _ in loaded]; durs = [d for _, d in loaded]
durs_a = np.array(durs)
print(f"n={len(audios)}", flush=True)

raw = {}
results = {"task": "R2 FlowSE per-N E_incr @ MAXN full-824", "idle_cold_mW": idle_cold_mW,
           "power_mode": nvp_q, "env_before": ENV_BEFORE, "per_N": {}}
for N in [1, 2, 3, 4, 5]:
    for y in audios[:3]:
        T_ = y.size(1); Y, _ = prep_Y(fse, y, device); fse_fn(Y, T_, N)()
    torch.cuda.synchronize()
    rec = TegrastatsRecorder(100); rec.start()
    lat, pw, L = [], [], []
    t_all = time.time()
    for y, d in zip(audios, durs):
        T_ = y.size(1); Y, _ = prep_Y(fse, y, device)
        ms, (ta, tb) = time_fwd_cuda(fse_fn(Y, T_, N))
        lat.append(ms); pw.append(rec.power_window(ta, tb)); L.append(tb - ta)
    rec.stop()
    lat = np.array(lat); pw_a = np.array(pw); L_a = np.array(L)
    rtf = lat / 1000 / durs_a
    E_cold = energy_per_utt(list(pw_a), list(L_a), durs, idle_cold_mW)
    raw[N] = {"pw": pw_a, "L": L_a}
    results["per_N"][f"N={N}"] = {
        "rtf_mean": float(rtf.mean()), "rtf_p95": float(np.percentile(rtf, 95)),
        "E_incr_cold_mJ_per_s": E_cold["E_incr_mJ_per_s_audio"],
        "E_total_mJ_per_s": E_cold["E_total_mJ_per_s_audio"],
        "P_load_mean_mW": float(np.nanmean(pw_a)), "wall_s": time.time() - t_all, "n": int(len(lat))}
    json.dump(results, open(OUT, "w"), indent=1)
    print(f"N={N} done: " + json.dumps(results['per_N'][f'N={N}']), flush=True)

rec_hot = TegrastatsRecorder(100); rec_hot.start(); time.sleep(60); rec_hot.stop()
idle_hot = rec_hot.idle_stats()
idle_hot_mW = idle_hot["vdd_mW"]["mean"] if (idle_hot and idle_hot.get("vdd_mW")) else None
results["idle_hot_mW"] = idle_hot_mW
for N in [1, 2, 3, 4, 5]:
    E_hot = energy_per_utt(list(raw[N]["pw"]), list(raw[N]["L"]), durs, idle_hot_mW)
    results["per_N"][f"N={N}"]["E_incr_hot_mJ_per_s"] = E_hot["E_incr_mJ_per_s_audio"]
results["env_after"] = env_snapshot("after")
json.dump(results, open(OUT, "w"), indent=1)
print("sha256:", hashlib.sha256(open(OUT, 'rb').read()).hexdigest(), flush=True)
print("ALL DONE", flush=True)
