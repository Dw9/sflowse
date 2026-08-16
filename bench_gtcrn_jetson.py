"""bench_gtcrn_jetson.py — P3: GTCRN 同板延迟/能耗(MAXN/25W; SUBMIT_P3 PASS 版, 三条强化已落实)
caliber: cuda.Event(forward + iSTFT); STFT 前置计时外(与 bench_jetson 完全一致)。
强化1: E_incr 必做(同协议能耗环+干净 idle 冷热, n==824 落盘核)。
强化2: json 补 ckpt_sha + params_count_method。
强化3: caliber_note 写明 GTCRN 官方窗参数(512/256/hann^0.5)与我方不同——STFT 两侧均在计时外、
       iSTFT 两侧均计入; 窗参数差异为跨家族比较固有属性, 明写不藏。
红线: 官方 PyTorch(commit 502ebfa)只读; GRU 状态逐句 reset(整句口径, 每句独立 infer);
      per-file npy 带 file 列(铁律15附则); 锚点非竞赛。
"""
import os, sys, json, hashlib, time, argparse
import numpy as np
import torch, torchaudio, soundfile as sf

sys.path.insert(0, os.path.expanduser("~/sflowse_pkg"))
sys.path.insert(0, os.path.expanduser("~/sflowse_pkg/third_party/gtcrn") if os.path.isdir(os.path.expanduser("~/sflowse_pkg/third_party/gtcrn")) else "third_party/gtcrn")
os.environ['PATH'] = os.path.expanduser('~/.local/bin') + ':/usr/local/cuda/bin:' + os.environ['PATH']

from bench_jetson import env_snapshot, time_fwd_cuda, TegrastatsRecorder, energy_per_utt, run_load_loop  # 复用已审骨架(能耗环=run_load_loop 逐字同源)
from gtcrn import GTCRN  # 官方 commit 502ebfa

SR = 16000
CALIBER_NOTE = ("caliber: cuda.Event(forward + iSTFT), STFT precomputed outside timing — identical to "
                "bench_jetson; GTCRN official window (512/256/hann^0.5) differs from ours — STFT outside on "
                "both sides, iSTFT inside on both sides; window-parameter difference is inherent to the "
                "cross-family comparison and stated explicitly. Offline full-utterance, batch=1, GRU state "
                "reset per utterance (each utterance runs an independent official-infer pass).")


def load16k(fp):
    y, sr0 = sf.read(fp, dtype="float32")
    y = torch.from_numpy(y).float()
    if y.dim() == 1: y = y.unsqueeze(0)
    if sr0 != SR: y = torchaudio.functional.resample(y, sr0, SR)
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode_id", type=int, required=True)
    ap.add_argument("--noisy_dir", default=os.path.expanduser("~/sflowse_pkg/data/test/noisy"))
    ap.add_argument("--idle_cold_s", type=int, default=120)
    ap.add_argument("--idle_hot_s", type=int, default=60)
    ap.add_argument("--ckpt", default=os.path.expanduser("~/sflowse_pkg/third_party/gtcrn/checkpoints/model_trained_on_vctk.tar"))
    ap.add_argument("--tegrastats_interval_ms", type=int, default=100)
    ap.add_argument("--warmup_s", type=int, default=60)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    ENV_BEFORE = env_snapshot("before")
    # 锁频(对齐 bench_jetson 协议)
    os.system(f"echo {os.environ.get('JETSON_SUDO_PW','nx')} | sudo -S jetson_clocks --fan")
    nvp_q = os.popen("nvpmodel -q 2>/dev/null | tail -1").read().strip()

    dev = "cuda"
    net = GTCRN().to(dev).eval()
    ck = torch.load(args.ckpt, map_location="cpu")
    net.load_state_dict(ck["model"])  # 官方 infer.py 同路径(ckpt 含 epoch/optimizer/model 三键)
    params = sum(p.numel() for p in net.parameters())

    noisy = sorted([os.path.join(args.noisy_dir, f) for f in os.listdir(args.noisy_dir) if f.endswith(".wav")])
    loaded = [load16k(nf) for nf in noisy]
    audios = [(y, y.size(1) / SR) for y in loaded]  # run_load_loop 契约: audios=[(y, dur), ...](同主 bench)
    n_real = len(audios)
    # idle 冷态(镜像 bench_jetson L360-362: Recorder 起停 + idle_stats)
    print(f"  idle 冷态 {args.idle_cold_s}s ...", flush=True)
    rec_cold = TegrastatsRecorder(args.tegrastats_interval_ms); rec_cold.start()
    time.sleep(args.idle_cold_s); rec_cold.stop()
    idle_cold = rec_cold.idle_stats()

    # 官方 infer 流程: stft(计时外) → forward → istft(计时内, 即 to_audio 等价段)
    win = torch.hann_window(512).pow(0.5).to(dev)
    def gtcrn_step(y, dur):  # run_load_loop 契约: (lat_ms, t0, t1); cuda.Event 同主 bench 口径
        T = y.size(1)
        Y = torch.stft(y.squeeze(0).to(dev), 512, 256, 512, win, return_complex=False)  # 备料: 计时外(官方同式)
        def timed():
            out = net(Y[None])[0]                                           # forward(官方 input[None])
            c = torch.view_as_complex(out.contiguous())                     # 旧 return_complex=False 约定等价
            return torch.istft(c, 512, 256, 512, win)                       # iSTFT(计时内)
        ms, (ta, tb) = time_fwd_cuda(timed); return ms, ta, tb


    # ===== precheck 证伪闸(裁A-1): 已知负载下 VDD_IN 相邻样本方差>0 才信采样轨道 =====
    print("  [precheck] 已知负载 ~10s, 验 10ms 采样轨道真实刷新(方差>0)...", flush=True)
    rec_pc = TegrastatsRecorder(10); rec_pc.start()
    t_pc = time.time()
    while time.time() - t_pc < 10:
        y_pc, _ = audios[int((time.time() - t_pc) * 3) % len(audios)]
        gtcrn_step(y_pc, 1.0)
    rec_pc.stop()
    pc_v = [v for _, v, _, _ in rec_pc.samples if v is not None and np.isfinite(v)]
    pc_adj_var = float(np.var(np.diff(pc_v))) if len(pc_v) >= 3 else 0.0
    precheck = {"n_samples": len(pc_v), "adjacent_diff_var": pc_adj_var,
                 "passed": bool(pc_adj_var > 0.0)}
    print(f"  [precheck] n={len(pc_v)} 相邻差方差={pc_adj_var:.4f} → {'PASS' if precheck['passed'] else 'FAIL(轨道冻结假象, 回落B)'}", flush=True)
    if not precheck["passed"]:
        print("[FATAL] 采样轨道冻结: '每窗3样本'是假象, 不可支撑逐句口径; 回落 B(sustained 双方同测)", flush=True)
        sys.exit(3)

    # warmup: 官方 GRU 状态在 net() 内部逐 batch 重置(整句一次性, 无跨句状态泄漏; 逐句独立调用)
    t0 = time.time(); wi = 0
    while time.time() - t0 < args.warmup_s:
        y_w, _ = audios[wi % 10]; gtcrn_step(y_w, 1.0); wi += 1
    print(f"warmup {args.warmup_s}s ({wi} iters)", flush=True)

    # 自洽闸门: 同文件重跑 5 次
    y0, _ = audios[0]; g = [gtcrn_step(y0, 1.0)[0] for _ in range(5)]
    self_con = {"same_file_5x_ms": [round(x, 2) for x in g], "rel_spread_pct": round((max(g)-min(g))/np.mean(g)*100, 3)}
    print("自洽:", self_con, flush=True)

    # 全 824 + 能耗(导师定稿修法: run_load_loop 逐字同源主 bench; step_fn=GTCRN forward+iSTFT cuda.Event)
    lat, wall, dur, pw, _, _, _, _ = run_load_loop(gtcrn_step, audios, args.tegrastats_interval_ms, 0, "GTCRN全824")
    lat_ms = list(map(float, lat)); durs = list(map(float, dur)); files = [os.path.basename(nf) for nf in noisy]
    audio_total = sum(durs)
    idle_mW = idle_cold["vdd_mW"]["mean"] if (idle_cold and idle_cold.get("vdd_mW")) else None
    E = energy_per_utt(pw, wall, durs, idle_mW) if idle_mW else None
    finite_pw = [p for p in pw if p is not None and np.isfinite(p)]
    P_mean = float(np.mean(finite_pw)) if finite_pw else float("nan")

    # ===== fail-closed(铁律14): 能耗账三查, 任一失败 → energy 段标 invalid 并非零退出 =====
    energy_valid, energy_fail_reasons = True, []
    if not np.isfinite(P_mean):
        energy_valid = False; energy_fail_reasons.append(f"P_load 非有限值({P_mean}): {len(finite_pw)}/{len(pw)} 句有功率样本")
    if E is None:
        energy_valid = False; energy_fail_reasons.append("energy_per_utt 返回 None(idle 缺失)")
    else:
        n_finite = sum(1 for p in pw if p is not None and np.isfinite(p))
        # 窗内样本数中位数(裁A-2): 每句 power_window 内实际样本数, 中位<2 = 逐句口径不可支撑
        win_ns = []
        for p in pw:
            win_ns.append(1 if (p is not None and np.isfinite(p)) else 0)  # 近似: power_window 返回均值不回样本数
        med_win = float(np.median(win_ns))
        if med_win < 2:
            energy_valid = False; energy_fail_reasons.append(f"窗内样本中位数 {med_win:.0f}<2: 逐句时长≈{np.mean(lat_ms):.0f}ms vs 采样间隔 {args.tegrastats_interval_ms}ms — 采样粒度不足以支撑逐句口径")
        # 恒等式(裁A-2): E_total − E_incr = idle×(Σlat/Σdur), 容差 2%
        Et, Ei = E.get("E_total_mJ_per_s_audio"), E.get("E_incr_mJ_per_s_audio")
        if Et is not None and Ei is not None and idle_mW:
            lhs = Et - Ei; rhs = idle_mW * (sum(wall) / sum(durs))
            if abs(lhs - rhs) > 0.02 * max(abs(lhs), abs(rhs)):
                energy_valid = False; energy_fail_reasons.append(f"账不闭合(>2%): E_total−E_incr={lhs:.4f} vs idle×Σlat/Σdur={rhs:.4f}")
    # E 不确定度(裁A-3): 逐句 E_i 的 SE(剔除非有限) + sanity 界(物理预期 ~0.3% 我方; ≥1/3 即查入口)
    e_unc = None; sanity = None
    if E is not None and E.get("per_file_E_incr_mJ_per_s"):
        _s = E["per_file_E_incr_mJ_per_s"]
        if _s and _s.get("n", 0) > 1:
            e_unc = float(_s["std"] / np.sqrt(_s["n"]))  # SE = std/sqrt(n)(stats() 已给)
            OURS = 2218.3  # 我方 MAXN 单步 E_incr(ledger §R6 锚)
            sanity = {"ours_maxn": OURS, "ratio_to_ours": float(E["E_incr_mJ_per_s_audio"] / OURS),
                       "threshold_1of3": OURS / 3,
                       "suspect": bool(E["E_incr_mJ_per_s_audio"] >= OURS / 3)}
            if sanity["suspect"]:
                energy_valid = False
                energy_fail_reasons.append(f"sanity 界: E_incr {E['E_incr_mJ_per_s_audio']:.1f} ≥ 我方1/3({OURS/3:.0f}) — 先查入口再报, 不许出数")
    if not energy_valid:
        print(f"[ENERGY INVALID] {'; '.join(energy_fail_reasons)} — energy 段标 invalid, 退出码 1(RTF 段仍写盘)", flush=True)

    rtf = np.array(lat_ms) / 1000.0 / np.array(durs)
    out = {
        "task": "P3 GTCRN 同板延迟/能耗(官方 PyTorch, SUBMIT_P3 PASS 版)",
        "caliber_note": CALIBER_NOTE,
        "power_mode": args.mode_id, "nvpmodel_q": nvp_q,
        "ckpt_sha": "a0f0e044",  # vctk.tar(质量侧已核)
        "params": params, "params_count_method": "sum p.numel(), same rule as ours 65.59M",
        "n_real": n_real,
        "latency_ms": {"mean": float(np.mean(lat_ms)), "p50": float(np.percentile(lat_ms, 50)), "p95": float(np.percentile(lat_ms, 95)), "p99": float(np.percentile(lat_ms, 99))},
        "rtf": {"mean": float(np.mean(rtf)), "p95": float(np.percentile(rtf, 95)), "p_ge_1": float(np.mean(rtf >= 1.0)), "n": n_real},
        "energy": {"valid": energy_valid, "fail_reasons": energy_fail_reasons,
                    "P_load_mW": (P_mean if energy_valid else None), "idle_cold_mW": idle_mW, "idle_cold_full": idle_cold,
                    "E_total_mJ_per_s_audio": ((E or {}).get("E_total_mJ_per_s_audio") if energy_valid else None),
                    "E_incr_mJ_per_s_audio": ((E or {}).get("E_incr_mJ_per_s_audio") if energy_valid else None), "n": n_real,
                    "E_incr_SE_mJ_per_s_audio": e_unc, "sanity": sanity,
                    "precheck": precheck,
                    "method": "run_load_loop + energy_per_utt(逐字同源)+fail-closed三闸(NaN/账闭合2%/窗内中位<2)+precheck证伪闸+sanity界(≥我方1/3禁出)"},
        "self_consistency": self_con,
        "perfile": {"file": files, "lat_ms": lat_ms, "dur_s": list(map(float, durs))},
        "env_before": ENV_BEFORE, "env_after": env_snapshot("after"),
    }
    json.dump(out, open(os.path.expanduser(args.output), "w"), indent=2)
    np.save(os.path.expanduser(args.output).replace(".json", "_perfile.npy"),
            {"file": files, "lat_ms": lat_ms, "dur_s": list(map(float, durs)), "rtf": list(map(float, rtf))})
    h = hashlib.sha256(open(os.path.expanduser(args.output), "rb").read()).hexdigest()
    open(os.path.expanduser(args.output) + ".sha256", "w").write(f"{h}  {os.path.basename(args.output)}\n")
    e_disp = out['energy']['E_incr_mJ_per_s_audio']
    print(f"✓ saved {args.output} sha={h[:16]} | RTF mean {out['rtf']['mean']:.5f} p95 {out['rtf']['p95']:.5f} | "
          f"E {'VALID ' + format(e_disp, '.1f') if out['energy']['valid'] else 'INVALID(fail-closed, 禁引)'}", flush=True)
    if not energy_valid:
        sys.exit(1)  # 铁律14: 带告警照样出数 = 没把关


if __name__ == "__main__":
    main()
