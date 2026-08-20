# [M54去重] 本文件与 archive 的 make_figs.py.m54-duplicate 功能等价(双agent并行产物); 以本文件为准(现行PNG/captions/build管线同源)。
"""make_figures.py — MAJOR-1 三张图(数字全从 json 读, 不手填; 显式源注记进图注)
Fig 6.1  RTF vs NFE 四档 + RTF=1 线(源: sweep v2×2 + v6×2)
Fig 6.2  漂移窗 RTF+温度+频率(源: v6 drift 段, 15W/10W; 锁频零漂移观测)
Fig 7.1  Pareto 质量-代价(质量=K-seed PESQ mean±std vs 设备代价=MAXN RTF(N); GTCRN 点=published PESQ + 实测 RTF)
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
OUTD = os.path.join(ROOT, "MDPI-PAPER", "figs")
os.makedirs(OUTD, exist_ok=True)

def load(rel):
    return json.load(open(os.path.join(ROOT, rel)))

def sha16(rel):
    import hashlib
    return hashlib.sha256(open(os.path.join(ROOT, rel), "rb").read()).hexdigest()[:16]

# ---------------- Fig 6.1 ----------------
SWEEPS = {"MAXN": "sweep_maxn_v2.json", "25W": "sweep_25w_v2.json",
          "15W": "bench_orin_15w_v6.json", "10W": "bench_orin_10w_v6.json"}
fig, ax = plt.subplots(figsize=(6.2, 4.2))
srcs = []
for label, rel in SWEEPS.items():
    d = load(rel); sw = d["flowse_nfe_sweep_cold"]
    ns = list(range(1, 7))
    mean = [sw[f"N={n}"]["rtf"]["mean"] for n in ns]
    p95 = [sw[f"N={n}"]["rtf"]["p95"] for n in ns]
    ax.plot(ns, mean, "o-", label=f"{label} (mean)")
    ax.plot(ns, p95, "o--", alpha=0.35, label=f"{label} (p95)")
    srcs.append(f"{rel}:{sha16(rel)[:8]}")
ax.axhline(1.0, color="k", lw=1, ls=":")
ax.text(6.05, 1.0, "RTF = 1", va="center", fontsize=9)
ax.set_xlabel("Solver steps N (NFE)"); ax.set_ylabel("RTF (offline, full-utterance)")
ax.set_xticks(range(1, 7)); ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
ax.set_title("RTF vs solver steps across power modes (clean sweeps)")
fig.tight_layout()
fig.savefig(os.path.join(OUTD, "fig6_1_rtf_nfe.png"), dpi=200)
print("Fig6.1 sources:", "; ".join(srcs))

# ---------------- Fig 6.2 ----------------
fig, axes = plt.subplots(2, 1, figsize=(6.2, 5.6), sharex=True)
srcs2 = []
axr = axes[0]; axt = axr.twinx()  # 唯一右轴(M57 结构修: 循环外建一次)
axr.set_ylabel("window RTF (per-file mean)"); axt.set_ylabel("temp (°C)")
axt.set_ylim(44, 48)  # M57: 两档温度共用刻度 44-48°C
for label, rel, col in (("15W", "bench_orin_15w_v6.json", "C2"), ("10W", "bench_orin_10w_v6.json", "C3")):
    d = load(rel)
    wins = [w for w in d["drift"]["windows_5min"] if not w.get("summary")]  # 尾项 summary 剔除
    idx = np.arange(len(wins))
    rtf = [w["rtf_mean"] for w in wins]; temp = [w["temp_C_mean"] for w in wins]; frq = [w["freq_MHz_mean"] for w in wins]
    # M55-3/M57: RTF(左)与温度(右)分轴; twinx 只建一次(循环外), 两档温度共轴共刻度, 杜绝双右轴叠印
    axr.plot(idx, rtf, "o-", color=col, label=f"{label} RTF")
    axt.plot(idx, temp, "s--", color=col, alpha=0.45, label=f"{label} temp")
    # M55-4: 频率 offset 计数(自 611 MHz)
    axes[1].plot(idx, [f - 611 for f in frq], "o-", color=col, label=f"{label} GPU clock − 611 MHz")
    srcs2.append(f"{rel}:{sha16(rel)[:8]}")
axes[0].grid(alpha=0.3)
h1, l1 = axr.get_legend_handles_labels(); h2, l2 = axt.get_legend_handles_labels()
axr.legend(h1 + h2, l1 + l2, fontsize=7, loc="center right")
axes[1].set_ylabel("GPU clock offset (MHz, from 611)")
axes[1].set_xlabel("5-min window index"); axes[1].legend(fontsize=7); axes[1].grid(alpha=0.3)
fig.suptitle("Load-window drift: RTF, temperature, GPU clock (locked)", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(OUTD, "fig6_2_drift.png"), dpi=200)
print("Fig6.2 sources:", "; ".join(srcs2))

# ---------------- Fig 7.1 ----------------
# 质量: K-seed PESQ (eval_dns/flowse_nfe_seeded_v2.json) N=1..5 + ResFlowSE m3(确定性)
# 代价: MAXN 干净冷扫 RTF(N)(sweep_maxn_v2.json) — N=6 用同曲线; ResFlowSE 单步 = maxn_ab_v3
seeded = load("eval_dns/flowse_nfe_seeded_v2.json")["curve"]
m3 = load("eval_dns/m3.json")
sweep = load("sweep_maxn_v2.json")["flowse_nfe_sweep_cold"]
ab = load("maxn_ab_v3.json")["resflowse_A"]["rtf"]["mean"]
gt = load("bench_gtcrn_maxn.json")
fig, ax = plt.subplots(figsize=(6.2, 4.4))
ns = list(range(1, 6))
pq = [seeded[f"N={n}"]["pesq_mean"] for n in ns]
pq_se = [seeded[f"N={n}"]["pesq_std"] for n in ns]
cost = [sweep[f"N={n}"]["rtf"]["mean"] for n in ns]
ax.errorbar(cost, pq, yerr=pq_se, fmt="o-", capsize=3, label="FlowSE (N=1..5; bars = K=3 seed std)")
m3_p = m3["metrics"]["pesq"]["mean"]
ax.plot([ab], [m3_p], "s", color="C3", label="M3 single-step (ours, deterministic)")
# GTCRN: 自测同管线 PESQ(eval_dns/gtcrn_vctk.json 直读, 不手填; M46 核过) + 同板实测 RTF
gtq = load("eval_dns/gtcrn_vctk.json")["measured_mean_std"]["pesq"]  # [mean, std]
# M55-1: GTCRN 确定性模型, 去误差棒(原 yerr=[gtq[1]] 为逐文件 std, 与 FlowSE 的 seed std 语义混口径)
ax.plot([gt["rtf"]["mean"]], [gtq[0]], "^", color="C0",
        label="GTCRN (self-measured, same pipeline; RTF measured; deterministic)")
ax.axhline(m3_p, color="C3", ls=":", lw=1, alpha=0.6)
ax.axvline(1.0, color="k", ls=":", lw=1)  # RTF=1 竖线(M55-2)
ax.annotate("N* = 4 (TOST, PESQ ±0.05)", xy=(cost[3], pq[3]), xytext=(cost[3]*1.3, pq[3]-0.08),
            arrowprops=dict(arrowstyle="->", lw=0.8), fontsize=9)
ax.set_xscale("log")
ax.set_xlabel("Cost: RTF at MAXN (offline, full-utterance; log scale)"); ax.set_ylabel("PESQ (VoiceBank-DEMAND test)")
ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
ax.set_title("Quality vs same-board cost (anchor comparison, not a contest)")
fig.tight_layout()
fig.savefig(os.path.join(OUTD, "fig7_1_pareto.png"), dpi=200)
print("Fig7.1 sources: flowse_nfe_seeded_v2.json, m3.json, sweep_maxn_v2.json, maxn_ab_v3.json, bench_gtcrn_maxn.json")

# 图注文件(显式源+sha 进 caption 供导师贴稿)
caps = {
 "fig6_1": "RTF vs solver steps at four power modes (clean sweeps, n=100 calibrated subsets; p95 dashed). Sources: " + "; ".join(srcs),
 "fig6_2": "Load-window (5 min) drift at 15W and 10W (clean ~60-min records): RTF (left axis) and temperature (right axis); bottom: GPU-clock offset from 611 MHz (constant within ±1 MHz = locked). No thermal degradation observed (observation, not proof); the MAXN 85-min longer record is discussed in §6.4 text. Sources: " + "; ".join(srcs2),
 "fig7_1": "Quality (PESQ) vs same-board cost (RTF at MAXN, log scale; vertical dotted line = RTF 1). FlowSE bars = K=3 seed std; GTCRN and ResFlowSE are deterministic (no seed variation, no bars). GTCRN point: self-measured PESQ 2.8543 (same pipeline, eval_dns/gtcrn_vctk.json; published 2.792 reported alongside in text) with on-device measured RTF; anchor, not a contest. Sources: eval_dns/flowse_nfe_seeded_v2.json; eval_dns/m3.json; eval_dns/gtcrn_vctk.json; sweep_maxn_v2.json; maxn_ab_v3.json; bench_gtcrn_maxn.json",
}
json.dump(caps, open(os.path.join(OUTD, "fig_captions.json"), "w"), indent=2)
print("✓ captions written; figs in", OUTD)
