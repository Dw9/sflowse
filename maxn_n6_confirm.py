"""maxn_n6_confirm.py — 任务#3(M36 扩容, 已批): MAXN 场次 N=6 全824 确认 + N=5 第二遍 + idle_hot 补录
复用 bench_jetson 骨架(prep_Y/TegrastatsRecorder/time_fwd_cuda/energy_per_utt/load_audio);
模型加载与 fse_fn 结构逐行镜像 bench_jetson L405-430(入口一致, 铁律15).
E 口径: idle_cold 用 v2 实测 8498.6 mW(json 落盘值, 非参数).
"""
import os, sys, json, hashlib, time, glob
import numpy as np, torch

sys.path.insert(0, os.path.expanduser("~/sflowse_pkg"))
os.environ['PATH'] = os.path.expanduser('~/.local/bin') + ':/usr/local/cuda/bin:' + os.environ['PATH']
from bench_jetson import env_snapshot, time_fwd_cuda, TegrastatsRecorder, load_audio, prep_Y, energy_per_utt
SR = 16000
IDLE_COLD_MW = 8498.6  # sweep_maxn_v2.json idle_cold 实测(json 落盘值)

ENV_BEFORE = env_snapshot("before")
os.system(f"echo {os.environ.get('JETSON_SUDO_PW','nx')} | sudo -S jetson_clocks --fan")
nvp_q = os.popen("nvpmodel -q 2>/dev/null | tail -1").read().strip()
print(f"power: {nvp_q} | cores: {os.cpu_count()}", flush=True)

device = "cuda"
# === 模型加载(镜像 bench_jetson L405-421) ===
from flowmse.data_module import SpecsDataModule
from flowmse.model import VFModel
from flowmse.sampling import get_white_box_solver
try:
    torch.serialization.add_safe_globals([SpecsDataModule])
except AttributeError:
    pass
fse = VFModel.load_from_checkpoint(os.path.expanduser("~/VB_DMD_FLOWSE_ICASSP_2025.ckpt"),
                                   base_dir=os.path.expanduser("~/sflowse_pkg/data"), map_location="cpu")
try:
    fse.data_module.setup(stage=None)
except Exception:
    pass
for name, p in fse.dnn.named_parameters():
    if name in fse.ema_dnn:
        p.data = fse.ema_dnn[name].to(p.device)
fse.cuda().eval()
T_rev, t_eps = fse.T_rev, fse.t_eps

def fse_fn(Y, T, N):  # 镜像 bench_jetson L424
    def _f():
        with torch.no_grad():
            sampler = get_white_box_solver("euler", fse.ode, fse, Y, T_rev=T_rev, t_eps=t_eps, N=N)
            sample, _ = sampler(); _ = fse.to_audio(sample.squeeze(), T)
    return _f

noisy = sorted(glob.glob(os.path.expanduser("~/sflowse_pkg/data/test/noisy/*.wav")))
loaded = [load_audio(nf) for nf in noisy]  # bench_jetson load_audio 返回 (y, dur)
audios = [y for y, _ in loaded]; durs = [d for _, d in loaded]
print(f"n={len(audios)}", flush=True)
for y in audios[:3]:
    T = y.size(1); Y, _ = prep_Y(fse, y, device); fse_fn(Y, T, 6)()

def full_pass(N, tag):
    lat, t0s, t1s, pws = [], [], [], []
    rec = TegrastatsRecorder(100); rec.start()
    for i, (y, d) in enumerate(zip(audios, durs)):
        T = y.size(1); Y, _ = prep_Y(fse, y, device)
        ms, (ta, tb) = time_fwd_cuda(fse_fn(Y, T, N))
        lat.append(ms); t0s.append(ta); t1s.append(tb); pws.append(rec.power_window(ta, tb))
        if (i+1) % 200 == 0:
            r = np.array(lat)/1000/np.array(durs[:len(lat)])
            print(f"  [{tag} {i+1}/{len(audios)}] RTF running {r.mean():.4f}", flush=True)
    rec.stop()
    rtf = np.array(lat)/1000/np.array(durs)
    E = energy_per_utt(pws, [b-a for a, b in zip(t0s, t1s)], durs, IDLE_COLD_MW)
    return {"rtf": {"mean": float(np.mean(rtf)), "p95": float(np.percentile(rtf, 95)),
                     "p99": float(np.percentile(rtf, 99)), "p_ge_1": float(np.mean(rtf >= 1.0)), "n": len(rtf)},
             "E_incr_coldidle_mJ_per_s_audio": (E or {}).get("E_incr_mJ_per_s_audio")}

out = {"task": "任务#3: MAXN N=6 全824 确认 + N=5 第二遍 + idle_hot 补录(M36 批)",
       "nvpmodel_q": nvp_q, "n": len(audios), "idle_cold_mW_src": "sweep_maxn_v2.json 8498.6",
       "N6": full_pass(6, "N6"), "N5": full_pass(5, "N5")}
rec_h = TegrastatsRecorder(100); rec_h.start(); time.sleep(120); rec_h.stop()
ih = rec_h.idle_stats()
out["idle_hot_120s"] = ({"vdd_mW_mean": ih["vdd_mW"]["mean"], "temp_C_mean": ih["temp_C"]["mean"]}
                          if ih and ih.get("vdd_mW") and ih["vdd_mW"].get("mean") is not None else None)
out["env_after"] = env_snapshot("after"); out["env_before"] = ENV_BEFORE
p = os.path.expanduser("~/maxn_n6_confirm.json")
json.dump(out, open(p, "w"), indent=2)
h = hashlib.sha256(open(p, "rb").read()).hexdigest()
open(p + ".sha256", "w").write(f"{h}  maxn_n6_confirm.json\n")
print("✓ saved", h[:16], json.dumps({k: out[k]["rtf"] for k in ("N6", "N5")}), flush=True)
