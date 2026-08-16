#!/usr/bin/env python3
"""外部机械检查:稿件里的实验数字是否都能在 RESULTS_LEDGER.md 找到。
不依赖任何 agent 的自觉——交付/汇报前跑一次。发现稿中出现、ledger 里没有的
数字 → 报警(可能是编造或未入账)。

用法: python check_numbers.py [manuscript.md ...]
默认检查 zh_manuscript_draft.md + draft_discussion_zh.md。
"""
import re, sys, os

LEDGER = "RESULTS_LEDGER.md"
# ⚠️ 2026-08-14 修:原清单里三个文件早已不存在(zh_manuscript_draft / draft_discussion_zh /
# en_manuscript_electronics),而**当前真正的稿件 draft_v2_measurement.md 从未被检查** ——
# 于是"交付前跑 check_numbers.py"这条纪律长期空转,只在检 cover_letter。
# 教训:本工具必须自己报告"哪些目标不存在",不能静默 skip 后打印"全部通过"。
DEFAULT_DOCS = ["MDPI-PAPER/draft_v2_measurement.md",   # ← 当前稿件,主检对象
                "MDPI-PAPER/cover_letter.md"]

# 关注"实验指标量级"的数字:PESQ~1-4, SI-SDR~10-20, 三位小数指标, 参数量M。
# 只抓像指标的数(带小数点、2-4位),放过年份/编号/引用号。
NUM_RE = re.compile(r"(?<![\d.])\d\.\d{2,6}(?![\d])")

# 作废上下文标记(铁律14: 带告警仍放行=没把关; 命中只在作废段 = 引用作废值)
VOID_MARK = re.compile(r"作废|撤回|污染|勘误|已结案, 附5|PENDING|无法回溯|不可引用|禁")

def load_ledger_lines(path):
    if not os.path.exists(path):
        print(f"[FATAL] 找不到 {path}"); sys.exit(2)
    lines = open(path, encoding="utf-8").readlines()
    return set(NUM_RE.findall("".join(lines))), lines

def clean_hit(num, ledger_lines, ledger_nums):
    """第二跳: num(或其近似命中 ±0.001, 与第一跳同口径)在 ledger 的每次出现,
    是否至少有一次不在作废上下文(行级判, 词边界)。近似命中必须落在 ledger 真实存在的数字上。"""
    try: v = float(num)
    except ValueError: return False
    def _dec(x): return len(x.split(".")[1]) if "." in x else 0
    cands = {num}
    for l in ledger_nums:
        try:
            # 近似仅允许"ledger 数精度 <= 稿数精度"(短对长), 防 0.999 借 ±0.001 窗口冒充 0.99991 的出处
            if abs(float(l) - v) <= 0.001 and _dec(l) >= _dec(num): cands.add(l)
        except ValueError: pass
    for c in cands:
        pat = re.compile(r"(?<![\d.])" + re.escape(c) + r"(?![\d])")
        for l in ledger_lines:
            if pat.search(l) and not VOID_MARK.search(l):
                return True
    return False

def approx_in(num, ledger_nums):
    """精确或差<=0.001(浮点/取整噪声)算命中; 同 clean_hit: 近似仅允许短精度对长精度。"""
    if num in ledger_nums:
        return True
    def _dec(x): return len(x.split(".")[1]) if "." in x else 0
    try:
        v = float(num)
        for l in ledger_nums:
            if abs(float(l) - v) <= 0.001 and _dec(l) >= _dec(num):
                return True
    except ValueError:
        pass
    return False

def main():
    docs = sys.argv[1:] or DEFAULT_DOCS
    ledger_nums, ledger_lines = load_ledger_lines(LEDGER)
    print(f"ledger 中指标数字 {len(ledger_nums)} 个: {sorted(ledger_nums)}\n")
    any_flag = False
    for doc in docs:
        if not os.path.exists(doc):
            print(f"[WARN] {doc} 不存在(清单失效——静默skip会让'全部通过'或虚假信号)"); any_flag = True; continue
        txt = open(doc, encoding="utf-8").read()
        flagged, void_only = [], []
        for m in NUM_RE.finditer(txt):
            num = m.group()
            line = txt[:m.start()].count("\n") + 1
            if not approx_in(num, ledger_nums):
                flagged.append((line, num))          # 第一跳: 查无来源
            elif not clean_hit(num, ledger_lines, ledger_nums):
                void_only.append((line, num))        # 第二跳: 只命中作废上下文
        if flagged or void_only:
            any_flag = True
            if flagged:
                print(f"[FAIL] {doc}: 以下数字在 ledger 中查无来源(可能编造/未入账):")
                for line, num in flagged:
                    print(f"    L{line}: {num}")
            if void_only:
                print(f"[FAIL] {doc}: 以下数字在 ledger 中仅出现于作废/勘误上下文(稿件在引作废值):")
                for line, num in void_only:
                    print(f"    L{line}: {num}")
        else:
            print(f"[OK] {doc}: 所有指标数字均可溯源到 ledger 的干净出处(第二跳过)。")
    print()
    if any_flag:
        print(">>> 有查无来源的数字。要么把它从源文件补进 RESULTS_LEDGER.md,要么删除/改'未测'。")
        sys.exit(1)
    print(">>> 全部通过:稿中数字均可溯源到 ledger。")

def selftest():
    """铁律14.2: 自失败测试 — 已知会失败的输入, 断言其确实失败(fail-closed 证明)。"""
    import tempfile, subprocess
    ok = True
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
        f.write("RTF 0.38083 与 0.99991\n"); doc = f.name
    r = subprocess.run([sys.executable, __file__, doc], capture_output=True, text=True)
    if r.returncode == 0:
        print("[SELFTEST FAIL] 作废值引用 + 无源数字 应非零退出, 实际 0"); ok = False
    else:
        print(f"[SELFTEST PASS] 已知坏输入 exit={r.returncode}(0.38083 二跳拦 / 0.99991 一跳拦)")
    r2 = subprocess.run([sys.executable, __file__, "/nonexistent_doc.md"], capture_output=True, text=True)
    if r2.returncode == 0:
        print("[SELFTEST FAIL] 目标缺失应非零退出, 实际 0"); ok = False
    else:
        print(f"[SELFTEST PASS] 目标缺失 exit={r2.returncode}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        main()
