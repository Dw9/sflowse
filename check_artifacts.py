#!/usr/bin/env python3
"""fail-closed 工件门禁(v4 §9.3 裁定,kimi R 闭环项 2):
  A. sidecar 完整性: 稿件 [SRC:] 引用的每个 json 在盘且带 .sha256, 且记录哈希 == 实测哈希;
     root 层全部 *.json 一并核(§6.1 的 "all artifacts carry SHA-256 sidecars" 断言按全集核)。
  B. 热学唯一源: 稿件四档漂移值 == 各 json drift.windows_5min 尾部 summary.first_last_rel_pct
     的两位小数舍入(唯一源纪律; 换源/换定义此处必红)。
用法: python3 check_artifacts.py [--selftest]
"""
import glob, hashlib, json, os, re, sys

MD = "MDPI-PAPER/draft_initial_review.md"
THERMAL = [("sweep_maxn_v2.json", 1.94), ("sweep_25w_v2.json", 0.90),
           ("bench_orin_15w_v6.json", 1.33), ("bench_orin_10w_v6.json", 2.13)]


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def check_sidecars():
    bad = []
    # [SRC:] 引用的 .npy 也入域(kimi v5 二刀要求): 必须在盘且带哈希核验过的 sidecar
    md_txt = open(MD, encoding="utf-8").read()
    import re as _re
    for m in _re.finditer(r"\[SRC:[^\]]*\]", md_txt):
        for f in _re.findall(r"([\w/*]+\.npy)", m.group()):
            if "*" in f:
                continue
            sc = f + ".sha256"
            if not os.path.exists(sc):
                bad.append(f"MISSING npy sidecar: {sc}")
            elif open(sc).read().split()[0] != sha256(f):
                bad.append(f"HASH MISMATCH (npy): {sc}")
    targets = sorted(glob.glob("*.json"))
    if not targets:
        return ["no json artifacts found at all"]
    for f in targets:
        sc = f + ".sha256"
        if not os.path.exists(sc):
            bad.append(f"MISSING sidecar: {sc}")
            continue
        rec = open(sc).read().split()[0]
        if rec != sha256(f):
            bad.append(f"HASH MISMATCH: {sc} records {rec[:16]}…, actual {sha256(f)[:16]}…")
    # 稿件 [SRC:] 引用的工件必须存在(裸名允许出现在子目录; * 为通配)
    md = open(MD, encoding="utf-8").read()
    for m in re.finditer(r"\[SRC:[^\]]*\]", md):
        for f in re.findall(r"([\w/*]+\.(?:json|npy|log|py|csv|md|tex))", m.group()):
            if f.startswith("ledger") or f.startswith("RESULTS_LEDGER") or f.startswith("refs_"):
                continue  # 账本/参考表类不在本门禁域
            if "*" in f:
                if not glob.glob(f):
                    bad.append(f"[SRC:] wildcard matches nothing: {f}")
                continue
            cands = [f, os.path.join("flowmse", f), os.path.join("MDPI-PAPER", f),
                     os.path.join("MDPI-PAPER", "figs", f)]
            if not any(os.path.exists(c) for c in cands):
                bad.append(f"[SRC:] cites missing file: {f}")
    return bad


def check_thermal():
    bad = []
    for f, want in THERMAL:
        try:
            d = json.load(open(f))
            summ = [w for w in d["drift"]["windows_5min"] if w.get("summary")]
            if not summ:
                bad.append(f"{f}: no drift summary entry")
                continue
            pct = summ[0]["first_last_rel_pct"]
            if abs(round(pct, 2) - want) > 0.005:
                bad.append(f"{f}: summary first_last_rel_pct={pct:.4f} "
                           f"(→{round(pct,2)}) != manuscript value {want}")
        except (KeyError, FileNotFoundError, json.JSONDecodeError) as e:
            bad.append(f"{f}: thermal source unreadable ({e})")
    return bad


def run():
    bad = check_sidecars() + check_thermal()
    if bad:
        print("[FAIL] artifact gate:")
        for b in bad:
            print("   ", b)
        return 1
    print(f"[ok] sidecars: {len(glob.glob('*.json'))} jsons all hashed & matching; "
          f"[SRC:] files exist; thermal 4-mode values match unique json summaries")
    return 0


def selftest():
    """fail-closed 自证: 造一个坏哈希 sidecar + 一个换定义的热学值, 断言门禁必红。"""
    import subprocess, tempfile
    ok = True
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, dir=".") as f:
        f.write('{"drift": {"windows_5min": [{"summary": true, "first_last_rel_pct": 99.9}]}}')
        tmp = f.name
    open(tmp + ".sha256", "w").write("0" * 64 + "  " + tmp)
    r = subprocess.run([sys.executable, __file__], capture_output=True, text=True)
    if r.returncode == 0:
        print("[SELFTEST FAIL] 坏哈希+假热学值应非零退出"); ok = False
    else:
        print(f"[SELFTEST PASS] 已知坏输入 exit={r.returncode}(哈希不匹配被拦)")
    # 热学支路单独自证: 把 THERMAL 指到假 json(99.9% vs 期望 1.94), check_thermal 必须报错
    import check_artifacts as self_mod
    self_mod.THERMAL = [(tmp, 1.94)]
    thermal_bad = self_mod.check_thermal()
    if thermal_bad:
        print("[SELFTEST PASS] 热学支路独立拦截:", thermal_bad[0][:60])
    else:
        print("[SELFTEST FAIL] 热学支路未拦住换源值"); ok = False
    os.unlink(tmp); os.unlink(tmp + ".sha256")
    return 0 if ok else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    sys.exit(run())
