"""sustained_bench.py — 裁B口径(导师定死 2026-08-16): 三族 sustained 能耗对照
GTCRN / ResFlowSE单步 / FlowSE N=1, x MAXN/25W 两档。
E_sust = (P_loop - idle_cold) × RTF_sustained, 带SE(P 窗样本 SE 传播)。
- 满载窗: 连跑 >=60s 无间隙(run_load_loop 已保证逐句无间隙); tegrastats 100ms(长窗多样本)。
- RTF_sustained = Σlat/Σdur(窗内); P_loop = 窗内 VDD_IN 样本均值。
- 主表我方逐句数不动; sustained 数只进 GTCRN 对照+明标 '不同caliber'。
- 三闸保持(NaN/账闭合/样本数)+precheck 不需要(长窗非逐句)。
"""
import os, sys, json, hashlib, time, glob
import numpy as np, torch
sys.path.insert(0, os.path.expanduser("~/sflowse_pkg"))
sys.path.insert(0, os.path.expanduser("~/sflowse_pkg/third_party/gtcrn"))
os.environ['PATH'] = os.path.expanduser('~/.local/bin') + ':/usr/local/cuda/bin:' + os.environ['PATH']
from bench_jetson import env_snapshot, time_fwd_cuda, TegrastatsRecorder, run_load_loop, load_audio, prep_Y, energy_per_utt
SR = 16000

MODE = sys.argv[1] if len(sys.argv) > 1 else None  # sustained_gtcrn | sustained_ours
OUT = os.path.expanduser(sys.argv[2]) if len(sys.argv) > 2 else None
assert MODE and OUT, "用法: sustained_bench.py <sustained_gtcrn|sustained_ours> <out.json>"

ENV_BEFORE = env_snapshot("before")
os.system(f"echo {os.environ.get('JETSON_SUDO_PW','nx')} | sudo -S jetson_clocks --fan")
nvp_q = os.popen("nvpmodel -q 2>/dev/null | tail -1").read().strip()
print(f"power: {nvp_q} | cores: {os.cpu_count()}", flush=True)
device = "cuda"
noisy = sorted(glob.glob(os.path.expanduser("~/sflowse_pkg/data/test/noisy/*.wav")))
audios = [load_audio(nf) for nf in noisy]  # 骨架 load_audio 已返回 (y, dur) — run_load_loop 契约
IDLE_COLD = {"0": 8498.6, "3": 7358.1}  # v2 族 idle_cold(json 落盘值)
idle_mW = IDLE_COLD.get(nvp_q)
assert idle_mW, f"无该档 idle_cold 记录(档 {nvp_q})"

def sustained(step_fn, tag, min_s=60):
    """连跑>=min_s 满载窗(run_load_loop drift_min 机制), 回 P_loop(SE)/RTF_sust/E_sust。"""
    lat, wall, dur, pw, _, _, samples, passes = run_load_loop(step_fn, audios, 100, max(1, int(min_s/((sum(d for _,d in audios)))))+1, tag)
    # 注意: drift_min 按墙钟; 直接传分钟数
    rtf_sust = float(np.sum(lat)/1000.0/np.sum(dur))
    sv = [v for _, v, _, _ in samples]
    sv = [v for v in sv if v is not None and np.isfinite(v)]
    if not sv:
        return {"valid": False, "fail": "无功率样本"}
    P_loop = float(np.mean(sv)); P_SE = float(np.std(sv, ddof=1)/np.sqrt(len(sv)))
    E_sust = (P_loop - idle_mW) * rtf_sust
    E_SE = P_SE * rtf_sust  # SE 传播(一阶)
    return {"valid": True, "wall_s": float(sum(wall)), "n_utt": len(lat), "passes": passes,
             "P_loop_mW": P_loop, "P_loop_SE": P_SE, "n_pow_samples": len(sv),
             "RTF_sustained": rtf_sust, "E_sust_mJ_per_s_audio": E_sust, "E_sust_SE": E_SE,
             "idle_cold_mW": idle_mW, "caliber": "sustained 满载窗(连跑>=60s, tegrastats 100ms), E=(P_loop-idle)×RTF_sust — 与逐句口径不同, 明标"}

out = {"task": f"裁B sustained({MODE})", "nvpmodel_q": nvp_q, "env_before": ENV_BEFORE}
if MODE == "sustained_gtcrn":
    from gtcrn import GTCRN
    net = GTCRN().to(device).eval()
    ck = torch.load(os.path.expanduser("~/sflowse_pkg/third_party/gtcrn/checkpoints/model_trained_on_vctk.tar"), map_location="cpu")
    net.load_state_dict(ck["model"]); out["ckpt_sha"] = "a0f0e044"; out["params"] = sum(p.numel() for p in net.parameters())
    win = torch.hann_window(512).pow(0.5).to(device)
    def g_step(y, dur):
        T = y.size(1)
        Y = torch.stft(y.squeeze(0).to(device), 512, 256, 512, win, return_complex=False)
        def timed():
            o = net(Y[None])[0]; c = torch.view_as_complex(o.contiguous()); return torch.istft(c, 512, 256, 512, win)
        ms, (ta, tb) = time_fwd_cuda(timed); return ms, ta, tb
    for y, _ in audios[:3]: g_step(y, 1.0)
    t0=time.time()
    while time.time()-t0 < 20: g_step(audios[int(time.time()*3)%10][0], 1.0)  # warmup
    # drift_min: 60s 负载 = 传 1(分钟, run_load_loop 按墙钟断)
    out["gtcrn_sustained"] = sustained(g_step, "GTCRN-sustained")
elif MODE == "sustained_ours":
    from flowmse.resflowse_model import ResFlowSEModel
    from flowmse.model import VFModel
    from flowmse.sampling import get_white_box_solver
    rfs = ResFlowSEModel.load_from_checkpoint(os.path.expanduser("~/sflowse.ckpt"), map_location="cpu", weights_only=False, strict=False)
    rfs.cuda().eval()
    from flowmse.data_module import SpecsDataModule
    try: torch.serialization.add_safe_globals([SpecsDataModule])
    except AttributeError: pass
    fse = VFModel.load_from_checkpoint(os.path.expanduser("~/VB_DMD_FLOWSE_ICASSP_2025.ckpt"), base_dir=os.path.expanduser("~/sflowse_pkg/data"), map_location="cpu")
    try: fse.data_module.setup(stage=None)
    except Exception: pass
    for name, p in fse.dnn.named_parameters():
        if name in fse.ema_dnn: p.data = fse.ema_dnn[name].to(p.device)
    fse.cuda().eval(); T_rev, t_eps = fse.T_rev, fse.t_eps
    def rfs_step(y, dur):
        T = y.size(1); Y, _ = prep_Y(rfs, y, device)
        def timed():
            with torch.no_grad():
                xh = rfs.forward(Y); _ = rfs.to_audio(xh.squeeze(), T)
        ms, (ta, tb) = time_fwd_cuda(timed); return ms, ta, tb
    def fse1_step(y, dur):
        T = y.size(1); Y, _ = prep_Y(fse, y, device)
        def timed():
            with torch.no_grad():
                s = get_white_box_solver("euler", fse.ode, fse, Y, T_rev=T_rev, t_eps=t_eps, N=1)
                sample, _ = s(); _ = fse.to_audio(sample.squeeze(), T)
        ms, (ta, tb) = time_fwd_cuda(timed); return ms, ta, tb
    for y, _ in audios[:3]: rfs_step(y, 1.0); fse1_step(y, 1.0)
    t0=time.time()
    while time.time()-t0 < 20: rfs_step(audios[int(time.time())%10][0], 1.0)
    out["resflowse_1nfe_sustained"] = sustained(rfs_step, "ResFlowSE单步-sustained")
    t0=time.time()
    while time.time()-t0 < 20: fse1_step(audios[int(time.time())%10][0], 1.0)
    out["flowse_n1_sustained"] = sustained(fse1_step, "FlowSE-N1-sustained")

out["env_after"] = env_snapshot("after")
json.dump(out, open(OUT, "w"), indent=2)
h = hashlib.sha256(open(OUT, "rb").read()).hexdigest()
open(OUT + ".sha256", "w").write(f"{h}  {os.path.basename(OUT)}\n")
for k in [k for k in out if k.endswith("sustained")]:
    v = out[k]
    print(f"✓ {k}: {'VALID' if v.get('valid') else 'INVALID'} RTF_sust {v.get('RTF_sustained',0):.5f} P {v.get('P_loop_mW',0):.0f}±{v.get('P_loop_SE',0):.0f}mW E_sust {v.get('E_sust_mJ_per_s_audio',0):.1f}±{v.get('E_sust_SE',0):.1f}", flush=True)
print(f"✓ saved {OUT} sha={h[:16]}", flush=True)
inv = [k for k in out if isinstance(out[k], dict) and out[k].get("valid") is False]
sys.exit(1 if inv else 0)
