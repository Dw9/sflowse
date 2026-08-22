"""n56_perfile.py — v5 M2 补测: MAXN N=5(x3遍)/N=6(x1遍) 全824 逐文件 RTF 数组落盘。
口径镜像 bench_jetson fse_fn+time_fwd_cuda(cuda.Event forward+iSTFT);每 N 前 3 句 warmup。"""
import os, sys, json, hashlib, time, glob
import numpy as np, torch
sys.path.insert(0, os.path.expanduser("~/sflowse_pkg"))
os.environ['PATH'] = os.path.expanduser('~/.local/bin') + ':/usr/local/cuda/bin:' + os.environ['PATH']
from bench_jetson import env_snapshot, time_fwd_cuda, load_audio, prep_Y

OUTJ = os.path.expanduser("~/n56_perfile.json")
OUTN = os.path.expanduser("~/n56_perfile.npy")
ENV_BEFORE = env_snapshot("before")
os.system(f"echo {os.environ.get('JETSON_SUDO_PW','nx')} | sudo -S jetson_clocks --fan")
device = "cuda"
nvp_q = os.popen("nvpmodel -q 2>/dev/null | tail -1").read().strip()
print(f"power: {nvp_q} | cores: {os.cpu_count()}", flush=True)
assert nvp_q.startswith("0") or "MAXN" in nvp_q.upper() or nvp_q == "0", "NOT MAXN: " + nvp_q

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
audios = [y for y, _ in loaded]; durs = np.array([d for _, d in loaded])
print(f"n={len(audios)}", flush=True)

results = {"task": "v5-M2 MAXN N=5(x3)/N=6(x1) full-824 per-file RTF arrays",
           "power_mode": nvp_q, "env_before": ENV_BEFORE, "per_N": {}}
arrays = {}
for N, passes in [(5, 3), (6, 1)]:
    for y in audios[:3]:
        T_ = y.size(1); Y, _ = prep_Y(fse, y, device); fse_fn(Y, T_, N)()
    torch.cuda.synchronize()
    all_lat = []
    for pi in range(passes):
        lat = []
        t0 = time.time()
        for y in audios:
            T_ = y.size(1); Y, _ = prep_Y(fse, y, device)
            ms, _ = time_fwd_cuda(fse_fn(Y, T_, N))
            lat.append(ms)
        all_lat.append(np.array(lat))
        print(f"N={N} pass{pi+1} done wall={time.time()-t0:.0f}s mean_rtf={np.mean(np.array(lat)/1000/durs):.5f}", flush=True)
    latcat = np.concatenate(all_lat); rtf = (latcat/1000/np.tile(durs, passes))
    arrays[f"N={N}"] = {"lat_ms": all_lat, "dur_s": durs.tolist()}
    results["per_N"][f"N={N}"] = {
        "passes": passes, "n_total": int(latcat.size),
        "rtf_mean": float(np.mean(rtf)), "rtf_p95": float(np.percentile(rtf, 95)),
        "rtf_p99": float(np.percentile(rtf, 99)),
        "P_rtf_ge1": float(np.mean(rtf >= 1.0)),
        "per_pass_mean_rtf": [float(np.mean(a/1000/durs)) for a in all_lat]}
    json.dump(results, open(OUTJ, "w"), indent=1)
    print(f"N={N} aggregate: " + json.dumps(results['per_N'][f'N={N}']), flush=True)

np.save(OUTN, arrays, allow_pickle=True)
results["env_after"] = env_snapshot("after")
json.dump(results, open(OUTJ, "w"), indent=1)
for f in (OUTJ, OUTN):
    print("sha256", os.path.basename(f), hashlib.sha256(open(f,'rb').read()).hexdigest(), flush=True)
print("ALL DONE", flush=True)
