"""a100_fp16_diag.py — 导师三问+对照: fp16 delta 参照物是否被 autocast 污染(2026-08-15)
嫌疑: 独立跑 json 无 fp32 指标; 若同进程 autocast 后的 fp32 也塌了, 则 |w32-w16|=1e-3 是'两个都塌的输出互差',
      与 fp32(干净进程)23.54dB 并不矛盾 —— 矛盾在参照物, 不在测量。
设计: 进程A: 每文件 autocast 先行(存wav) → fp32-after(存wav); 进程B(独立, 无 autocast): fp32-clean(存wav)。
指标: 各输出 vs clean(PESQ/SI-SDR) + 关键三对差: fp32clean↔autocast(真delta), fp32after↔autocast(旧delta来源),
      fp32clean↔fp32after(顺序污染量); delta 全部在 to_audio 之后、同一 norm 下盘上 wav 重算。
"""
import os, sys, json, hashlib
os.environ.setdefault("NCSNPP_PURE_PYTORCH", "1"); os.environ.setdefault("NCSNPP_PURE_UPFIRDN", "1")
import subprocess, numpy as np, torch, soundfile as sf
from pesq import pesq
from flowmse.util.other import si_sdr
SR = 16000
FILES = ["p232_001", "p232_326", "p257_434"]
BASE = "/home/zhibo/workspace/VoiceBank_processed/test"

def enhance(model, y, T, autocast):
    yn = y / y.abs().max()
    from flowmse.util.other import pad_spec
    Y = pad_spec(torch.unsqueeze(model._forward_transform(model._stft(yn.cuda())), 0))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        xh = model.forward(Y)
    xh = xh.to(torch.complex64)
    return model.to_audio(xh.squeeze(), T).float().cpu().numpy()

def enhance32(model, y, T):
    yn = y / y.abs().max()
    from flowmse.util.other import pad_spec
    Y = pad_spec(torch.unsqueeze(model._forward_transform(model._stft(yn.cuda())), 0))
    with torch.no_grad():
        xh = model.forward(Y)
    return model.to_audio(xh.squeeze(), T).float().cpu().numpy()

# ---- 进程A: autocast 先行, 后 fp32-after ----
def run_A():
    from flowmse.resflowse_model import ResFlowSEModel
    m = ResFlowSEModel.load_from_checkpoint("sflowse.ckpt", map_location="cpu", weights_only=False, strict=False)
    m.cuda().eval()
    for bn in FILES:
        y, _ = sf.read(f"{BASE}/noisy/{bn}.wav", dtype="float32"); y = torch.from_numpy(y).unsqueeze(0)
        T = y.size(1)
        sf.write(f"diagA_{bn}_autocast.wav", enhance(m, y, T, True), SR, subtype="FLOAT")
        sf.write(f"diagA_{bn}_fp32after.wav", enhance32(m, y, T), SR, subtype="FLOAT")
        print(f"A done {bn}", flush=True)

# ---- 进程B: 独立, 只有 fp32, 无 autocast ----
def run_B():
    from flowmse.resflowse_model import ResFlowSEModel
    m = ResFlowSEModel.load_from_checkpoint("sflowse.ckpt", map_location="cpu", weights_only=False, strict=False)
    m.cuda().eval()
    for bn in FILES:
        y, _ = sf.read(f"{BASE}/noisy/{bn}.wav", dtype="float32"); y = torch.from_numpy(y).unsqueeze(0)
        T = y.size(1)
        sf.write(f"diagB_{bn}_fp32clean.wav", enhance32(m, y, T), SR, subtype="FLOAT")
        print(f"B done {bn}", flush=True)

def metrics(ref, x):
    n = min(len(ref), len(x)); a, b = ref[:n], x[:n]
    return {"pesq": round(float(pesq(SR, a, b, "wb")), 3), "si_sdr": round(float(si_sdr(a, b)), 3)}

def delta(a_path, b_path):
    a, _ = sf.read(a_path, dtype="float32"); b, _ = sf.read(b_path, dtype="float32")
    n = min(len(a), len(b)); a, b = a[:n], b[:n]
    d = np.abs(a - b)
    return {"max_abs": float(d.max()), "rms_ratio_pct": float(np.sqrt((d**2).mean()) / (np.sqrt((a**2).mean()) + 1e-12) * 100)}

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "A": run_A(); sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1] == "B": run_B(); sys.exit(0)
    # 调度两个独立进程
    for tag in ("A", "B"):
        r = subprocess.run([sys.executable, __file__, tag], capture_output=True, text=True)
        print(r.stdout[-500:], r.stderr[-300:] if r.returncode else "", flush=True)
    # 盘上重算(导师(1): 直接在盘上重算 max|Δ| 与 RMS 比)
    out = {"task": "fp16 delta 参照物诊断(导师三问)", "files": FILES,
           "delta_where": "to_audio 之后/两路同一 norm(yn=y/max|y|)/盘上 wav 重算",
           "per_file": {}}
    for bn in FILES:
        clean, _ = sf.read(f"{BASE}/clean/{bn}.wav", dtype="float32")
        rec = {}
        for tag, path in [("autocast", f"diagA_{bn}_autocast.wav"), ("fp32after", f"diagA_{bn}_fp32after.wav"), ("fp32clean", f"diagB_{bn}_fp32clean.wav")]:
            w, _ = sf.read(path, dtype="float32"); rec[tag] = {"vs_clean": metrics(clean, w)}
        rec["d_fp32clean_vs_autocast"] = delta(f"diagB_{bn}_fp32clean.wav", f"diagA_{bn}_autocast.wav")
        rec["d_fp32after_vs_autocast"] = delta(f"diagA_{bn}_fp32after.wav", f"diagA_{bn}_autocast.wav")
        rec["d_fp32clean_vs_fp32after"] = delta(f"diagB_{bn}_fp32clean.wav", f"diagA_{bn}_fp32after.wav")
        # 导师(3): fp16(autocast) vs fp32-clean 之间的 SI-SDR
        a, _ = sf.read(f"diagB_{bn}_fp32clean.wav", dtype="float32"); c, _ = sf.read(f"diagA_{bn}_autocast.wav", dtype="float32")
        n = min(len(a), len(c)); rec["si_sdr_fp32clean_vs_autocast"] = round(float(si_sdr(a[:n], c[:n])), 3)
        out["per_file"][bn] = rec
        print(bn, json.dumps(rec), flush=True)
    json.dump(out, open("a100_fp16_diag.json", "w"), indent=2, ensure_ascii=False)
    sha = hashlib.sha256(open("a100_fp16_diag.json", "rb").read()).hexdigest()
    open("a100_fp16_diag.json.sha256", "w").write(f"{sha}  a100_fp16_diag.json\n")
    print(f"✓ saved a100_fp16_diag.json sha={sha[:16]}", flush=True)
