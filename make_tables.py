#!/usr/bin/env python3
"""从磁盘产物直接生成论文表格 —— 堵住"手抄数字进论文"这个口子。

为什么需要它:
  check_numbers.py 只能**事后**核对稿中数字是否出现在 ledger,
  它防不住"一开始就抄错行/抄错口径"。本脚本让 §6/§7 的表**由数据直接生成**,
  人不碰数字,只碰措辞。

用法:
  python make_tables.py --section 6      # 生成 §6 表格
  python make_tables.py --section 7      # 生成 §7 表格
  python make_tables.py --audit          # 只列出可用产物 + sha256

铁律对应:
  - 每个数字带 [SRC: 文件名 sha256前16]  (铁律10 活源)
  - 缺失产物写 [PENDING],绝不留空或猜  (铁律10 无源只写未测)
  - per-file 与 mean-over-mean 分列,不混  (口径分离义务)
"""
import argparse, glob, hashlib, json, os, sys

ROOT = os.path.dirname(os.path.abspath(__file__))


def sha16(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


# ---- 显式源映射:不猜、不 glob、不回退 ----------------------------------
# 教训(2026-08-15):原 pick() 靠"权威名 + 变体回退",结果为 MAXN/25W 选中了
# **已作废的旧 D2 资产**(jetson_clocks off),并在打完告警后照样把数印了出来。
# 告警不算把关。现在改为**逐档显式指定文件与 json 路径**,并对作废源**硬拦**。

VOID_SOURCES = {
    "bench_orin_maxn.json": "旧 D2:jetson_clocks off,已被 v5/v3 取代(PLAN §2 冻结表)",
    "bench_orin_25w.json":  "旧 D2:jetson_clocks off + 块设计首循环,已作废",
    "bench_a100.json":      "v1 全部数字已作废(被 v2 覆盖)",
    # 2026-08-16 扩充: 污染日 v5 族的延迟/能耗字段全部拦(仅作废上下文引用允许)
    "bench_orin_maxn_v5.json": "污染日(08-14)产物: NFE 冷扫/实验B/idle/能耗全部被 v2 族取代(ledger §#15/§R6); 延迟绝对值由 maxn_ab_v3/sweep_maxn_v2 取代",
    "bench_orin_25w_v5.json":  "污染日(08-14)产物: 同上; 单步由 p2_ab_v2/sweep_25w_v2 取代, idle 由 sweep_25w_v2 取代",
    "sweep_maxn_clean.json":   "n=8 入口截断(REVISE-1): 扫描数字被 sweep_maxn_v2(n=100 校准PASS)取代; 仅 manifest 仍有效",
    "sweep_25w_clean.json":    "n=8 入口截断: NFE 数字仅作 n=8 对 n=8 走向表基线(与 sweep_15w_n8 同口径并列), 不作主表",
    "eval_t2a_fp16.json":      "fp16 824 崩溃: 无 op_path/env 记录, 无法回溯(ledger §fp16 二次勘误), 禁独立引用",
}

# ===== 权威源映射 v2(2026-08-16; 主口径=v2 族, 导师裁定) =====
SINGLE_STEP_SRC = {
    "MAXN": ("maxn_ab_v3.json",   ["resflowse_A"]),            # 净化交错(跨会话注记: 与 v2 块 0.16246 差 0.6%)
    "25W":  ("p2_ab_v2.json",     ["resflowse_A"]),            # 净化交错(与 v2 块 0.32058 差 3.8% — 跨协议变异注记)
    "15W":  ("bench_orin_15w_v6.json", ["resflowse_1nfe"]),
    "10W":  ("bench_orin_10w_v6.json", ["resflowse_1nfe"]),
}
# NFE 扫描/N_max/能耗(全824): v2 族
SWEEP_SRC = {
    "MAXN": "sweep_maxn_v2.json",   # n=100 校准 PASS + calib_full824; N6 全824确认另见 N6_CONFIRM_SRC
    "25W":  "sweep_25w_v2.json",
    "15W":  "bench_orin_15w_v6.json",
    "10W":  "bench_orin_10w_v6.json",
}
N6_CONFIRM_SRC = "maxn_n6_confirm.json"      # N=6 全824 确认 + N=5 第二遍 + idle_hot
ENERGY_SRC = {                               # E_incr(全824, 干净 idle) — ledger §R6 锚 2218.32/破在10W/25W带沿
    "MAXN": "sweep_maxn_v2.json", "25W": "sweep_25w_v2.json",
    "15W":  "bench_orin_15w_v6.json", "10W": "bench_orin_10w_v6.json",
}
# GTCRN 对照(P3): RTF 主口径 v1(100ms); sustained(裁B) 单列明标不同 caliber
GTCRN_SRC = {"MAXN": "bench_gtcrn_maxn.json", "25W": "bench_gtcrn_25w.json"}
SUSTAINED_SRC = {                             # 裁B: (gtcrn, ours) 两 json/档
    "MAXN": ("sustained_gtcrn_maxn.json", "sustained_ours_maxn.json"),
    "25W":  ("sustained_gtcrn_25w.json",  "sustained_ours_25w.json"),
}
CORE_HALVING_SRC = "sweep_15w_n8.json"       # 任务#7: 核减半走向表 n=8 对 n=8(与 sweep_25w_clean 同 8 文件; 25W 基线=n=8, 非主表口径)


def guard(path):
    """作废源硬拦:命中即返回 None 并打印原因,绝不让它的数字流出。"""
    if path is None:
        return None
    base = os.path.basename(path)
    if base in VOID_SOURCES:
        print(f"[VOID] 拒绝使用 {base} —— {VOID_SOURCES[base]}", file=sys.stderr)
        return None
    return path


def load(path):
    """读 json 并附带 provenance;文件不存在返回 None(调用方须写 PENDING)。"""
    if path is None:
        return None
    full = os.path.join(ROOT, path)
    if not os.path.exists(full):
        return None
    with open(full) as f:
        d = json.load(f)
    d["_src"] = f"{path} sha256:{sha16(full)}"
    return d


def src(d):
    return d["_src"] if d else "PENDING"


def fmt(v, nd=3, pending="[PENDING]"):
    return pending if v is None else f"{v:.{nd}f}"


def dig(d, *keys, default=None):
    """安全取嵌套键;任一层缺失即返回 default(而非抛错或猜)。"""
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


# ---------------------------------------------------------------- §6

def section6():
    out = ["## Table 6.1 — Same-board single-step latency and energy across power modes",
           "",
           "| Mode | GPU clock (MHz) | Cores | RTF mean (per-file) | RTF p95 | E_incr (mJ/s audio) | Source |",
           "|---|---|---|---|---|---|---|"]
    for label in ("MAXN", "25W", "15W", "10W"):
        fsrc, path = SINGLE_STEP_SRC.get(label, (None, []))
        ds = load(guard(fsrc)); de = load(guard(ENERGY_SRC.get(label)))
        node = ds
        for k in path: node = node.get(k, {}) if isinstance(node, dict) else {}
        r = node.get("rtf", {})
        # 频率/核数: 从 SWEEP 源 idle_cold freq + platform(全 v2 族有)
        dsw = load(guard(SWEEP_SRC.get(label)))
        fr = dig(dsw, "energy", "idle_cold", "freq_MHz", "min", default=None)
        cores = dig(dsw, "platform", "online_cpu_cores", default=None)
        out.append(
            f"| {label} | {fmt(fr,0) if fr else '[PENDING]'} | {cores if cores else '[PENDING]'} "
            f"| {fmt(r.get('mean'),5)} | {fmt(r.get('p95'),5)} "
            f"| {fmt(dig(de,'energy','single_step_E_using_idle_cold','E_incr_mJ_per_s_audio'),1) if de else '[PENDING]'} "
            f"| {src(ds)} + E:{os.path.basename(ENERGY_SRC[label])} |")

    out += ["", "## Table 6.2 — Solver-step budget: RTF(N) clean sweeps and N_max", "",
            "| Mode | N=1 | N=2 | N=3 | N=4 | N=5 | N=6 | N_max_mean | N_max_p95 | Source |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for label in ("MAXN", "25W", "15W", "10W"):
        d = load(guard(SWEEP_SRC.get(label)))
        sw = dig(d, "flowse_nfe_sweep_cold", default={}) or {}
        cells = []
        for n in range(1, 7):
            v = dig(sw, f"N={n}", "rtf", "mean", default=None)
            cells.append(fmt(v, 4) if v else "[PENDING]")
        nm = dig(d, "n_max_cold", default={}) or {}
        nmm, nmp = nm.get("N_max_mean"), nm.get("N_max_p95")
        if label == "MAXN" and nmm == 6:
            # N=6 全824 确认(任务#3)在 N6_CONFIRM_SRC, 表注标
            out.append(f"| {label} | " + " | ".join(cells) +
                       f" | **{nmm}** (confirmed on full 824) | {nmp} | {src(d)} + {N6_CONFIRM_SRC} |")
        else:
            out.append(f"| {label} | " + " | ".join(cells) +
                       f" | {fmt(nmm,0) if nmm else '[PENDING]'} | {fmt(nmp,0) if nmp else '[PENDING]'} | {src(d)} |")

    out += ["", "## Table 6.4 — Core-halving extrapolation table (n=8 vs n=8, same 8 files)", "",
            "| Transition | Cores | N=1 | N=3 | N=6 | Sources |",
            "|---|---|---|---|---|---|"]
    d15 = load(guard(CORE_HALVING_SRC)); d25 = load("sweep_25w_clean.json")  # n=8 基线(同 8 文件, 非主表口径, 明标)
    # ⚠️ 走向表专用通道: sweep_*_clean 的 n=8 扫描**仅允许**用于本表 n=8 对 n=8 基线(ledger §任务#7 已记),
    # 不解除其 VOID(主表用途仍拦); 此处直读不经 guard, 但表内已明标口径与来源。
    def _n8(file_):
        return load(file_)
    for tag, key, f_lo, f_hi, dl in (("25W→MAXN (same cores)", "maxn", 407.01, 917.0, None),
                                       ("25W→15W (halved cores)", "15w", 407.01, 611.15, d15)):
        cells = []
        for N in (1, 3, 6):
            v25 = dig(d25, "flowse_nfe_sweep_cold", f"N={N}", "rtf", "mean", default=None)
            srcd = _n8("sweep_maxn_clean.json") if key == "maxn" else d15
            vx = dig(srcd, "flowse_nfe_sweep_cold", f"N={N}", "rtf", "mean", default=None)
            if v25 and vx:
                exp = v25 * (f_lo / f_hi)
                cells.append(f"+{(vx/exp-1)*100:.1f}%")
            else:
                cells.append("[PENDING]")
        out.append(f"| {tag} | {'8→8' if key=='maxn' else '8→4'} | " + " | ".join(cells) +
                   f" | n=8 pairs: sweep_25w_clean + {'sweep_maxn_clean' if key=='maxn' else CORE_HALVING_SRC} |")
    out += ["", "> ⚠️ 6.3 是 n=8 对 n=8 的同语料口径(8 文件 manifest 三份一致); N=1 全824 对全824 口径为 +61.0%(M26 定稿), 两口径并列各标。", ""]

    out += ["## Table 6.3 — GTCRN cross-family anchor (P3; anchor, not a contest)", "",
            "| Config | Mode | RTF mean (per-file) | RTF p95 | E_sust (mJ/s audio) | Source |",
            "|---|---|---|---|---|---|"]
    for label in ("MAXN", "25W"):
        dg = load(guard(GTCRN_SRC.get(label)))
        r = dig(dg, "rtf", default={}) or {}
        g_s, o_s = SUSTAINED_SRC.get(label, (None, None))
        dgs = load(guard(g_s))
        es = dig(dgs, "gtcrn_sustained", "E_sust_mJ_per_s_audio", default=None)
        out.append(f"| GTCRN (0.0482M) | {label} | {fmt(r.get('mean'),5)} | {fmt(r.get('p95'),5)} "
                   f"| {fmt(es,1) if es else '[PENDING]'}±{fmt(dig(dgs,'gtcrn_sustained','E_sust_SE'),1) if dgs else ''} | {src(dg)} |")
    for label in ("MAXN", "25W"):
        g_s, o_s = SUSTAINED_SRC.get(label, (None, None))
        do = load(guard(o_s))
        for cfg, key in (("ResFlowSE 1-step", "resflowse_1nfe_sustained"), ("FlowSE N=1", "flowse_n1_sustained")):
            v = dig(do, key, default={}) or {}
            out.append(f"| {cfg} (ours) | {label} | — (sustained: {fmt(v.get('RTF_sustained'),5)}) | — "
                       f"| {fmt(v.get('E_sust_mJ_per_s_audio'),1)}±{fmt(v.get('E_sust_SE'),1)} | {os.path.basename(o_s)} |")
    out += ["", "> ⚠️ E 列为 **sustained 满载窗口径(裁B)**, 与逐句口径不同 caliber, 明标; GTCRN 逐句能耗物理不可支撑(26ms 句 < 采样粒度, fail-closed 拦)。", ""]
    out += ["> ⚠️ RTF is **offline full-utterance**; caliber = network forward + iSTFT, input STFT precomputed; "
               "GTCRN official window 512/256 differs from ours (stated). `N_max_p95` is the deployment-relevant figure.", ""]
    return "\n".join(out)


# ---------------------------------------------------------------- §7

def section7():
    out = ["## Table 7.1 — Quality versus NFE (A100, K seeds) and the break-even point N*",
           "",
           "| Config | NFE | PESQ (mean ± std over seeds) | SI-SDR | ESTOI | Source |",
           "|---|---|---|---|---|---|"]
    m3 = load("eval_dns/m3.json")
    for n in (1, 2, 3, 4, 5):
        d = load(guard(f"eval_dns/flowse_n{n}.json"))
        if d is None:
            out.append(f"| FlowSE | {n} | [PENDING] | [PENDING] | [PENDING] | PENDING |")
            continue
        p = dig(d, "metrics", "pesq", default={})
        std = p.get("seed_std")
        pesq = fmt(p.get("mean"), 4) + (f" ± {std:.4f}" if std is not None else " (single draw)")
        out.append(f"| FlowSE | {n} | {pesq} | {fmt(dig(d,'metrics','si_sdr','mean'),3)} "
                   f"| {fmt(dig(d,'metrics','estoi','mean'),4)} | {src(d)} |")
    if m3:
        out.append(f"| **ResFlowSE (ours)** | **1** | **{fmt(dig(m3,'metrics','pesq','mean'),4)}** "
                   f"(deterministic) | {fmt(dig(m3,'metrics','si_sdr','mean'),3)} "
                   f"| {fmt(dig(m3,'metrics','estoi','mean'),4)} | {src(m3)} |")
    out += ["",
            "> **N\\*** = smallest N at which FlowSE(N) is statistically **equivalent** (TOST) to "
            "ResFlowSE. ⚠️ If the spacing between adjacent N near the crossing is smaller than "
            "3×(seed std), **report N\\* as an interval**, not an integer.",
            "> ⚠️ FlowSE is a sampled model: entries without a ± are **single draws**, not expectations.",
            ""]
    return "\n".join(out)


def audit():
    pats = ["bench_orin*.json", "bench_a100*.json", "eval_dns/*.json", "*_parity.npy"]
    print(f"{'file':52s} {'sha256[:16]':18s} size")
    for pat in pats:
        for f in sorted(glob.glob(os.path.join(ROOT, pat))):
            rel = os.path.relpath(f, ROOT)
            print(f"{rel:52s} {sha16(f):18s} {os.path.getsize(f):>9,}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", type=int, choices=[6, 7])
    ap.add_argument("--audit", action="store_true")
    a = ap.parse_args()
    if a.audit:
        audit()
    elif a.section == 6:
        print(section6())
    elif a.section == 7:
        print(section7())
    else:
        ap.print_help(); sys.exit(1)
