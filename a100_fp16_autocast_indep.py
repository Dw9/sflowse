"""a100_fp16_autocast_indep.py — fp16 autocast 独立跑补 sha 产物(导师 2026-08-15 派活)
设计: autocast + pure 回退算子(NCSNPP_PURE_PYTORCH/UPFIRDN), **每条 autocast 先行**(无 fp32 预跑,
     针对此前谜团: autocast 独立跑曾复现摧毁(PESQ 1.66), 但后续独立跑 10/10 clean, fp32 先行也 clean);
     fp32 仅在 autocast 之后跑, 作为 max|delta| 的事后参考(不构成 autocast 的前置条件)。
字段: k_of_10_ran_clean / k_of_10_collapsed(禁用 k_of_10_reproduced — 会被误读)
顺手: 打印 buffer dtype(搭便车不额外占时间)。
不报数字入稿: 本文件是 fp16 机制收尾的落盘源, §6.5 只引用其"响亮失败/静默通过"的定性结论。
"""
import os, json, hashlib
from pathlib import Path
os.environ.setdefault("NCSNPP_PURE_PYTORCH", "1")
os.environ.setdefault("NCSNPP_PURE_UPFIRDN", "1")
import numpy as np
import torch, soundfile as sf
from pesq import pesq
from pystoi import stoi
SR = 16000
from flowmse.util.other import si_sdr, pad_spec

def load_audio(p):
    x, _ = sf.read(p, dtype="float32")
    return torch.from_numpy(x).unsqueeze(0)

def spec(x):  # complex64 STFT 与模型入口一致
    from flowmse.util.other import pad_spec
    stft = torch.stft(x.squeeze(0), n_fft=510, hop_length=128, win_length=32 * 16,
                      window=torch.hann_window(32 * 16, device=x.device), return_complex=True,
                      center=True, pad_mode="reflect")
    return pad_spec(torch.unsqueeze(stft, 0))

def enhance_auto_first(model, y, T):
    yn = y / y.abs().max()
    Y = pad_spec(torch.unsqueeze(model._forward_transform(model._stft(yn.cuda())), 0))
    # ① autocast 先行(pure 算子)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        xh16 = model.forward(Y)
    dtypes = {n: str(b.dtype) for n, b in model.named_buffers() if b.dtype in (torch.float16, torch.bfloat16, torch.complex32)}
    xh16 = xh16.to(torch.complex64)
    w16 = model.to_audio(xh16.squeeze(), T).float().cpu().numpy()
    # ② fp32 事后参考(delta 基准, 非 autocast 前置)
    with torch.no_grad():
        xh32 = model.forward(Y)
    w32 = model.to_audio(xh32.squeeze(), T).float().cpu().numpy()
    return w32, w16, xh32.squeeze().float().cpu().numpy(), xh16.squeeze().float().cpu().numpy(), dtypes

def main():
    from flowmse.resflowse_model import ResFlowSEModel
    rfs = ResFlowSEModel.load_from_checkpoint("sflowse.ckpt", map_location="cpu", weights_only=False, strict=False)
    rfs.cuda().eval()
    clean_all = sorted(str(p) for p in Path("/home/zhibo/workspace/VoiceBank_processed/test/clean").glob("*.wav"))
    noisy_all = sorted(str(p) for p in Path("/home/zhibo/workspace/VoiceBank_processed/test/noisy").glob("*.wav"))
    idx = [0] + [int(v) for v in np.linspace(1, len(clean_all) - 1, 9)]
    files = [(clean_all[i], noisy_all[i]) for i in idx]

    results, n_clean, n_collapsed = [], 0, 0
    for cf, nf in files:
        bn = os.path.basename(nf); rec = {"file": bn}
        x = load_audio(cf); y = load_audio(nf); T = y.size(1); xn = x.squeeze().numpy()
        try:
            w32, w16, s32, s16, dtypes = enhance_auto_first(rfs, y, T)
            m = min(len(xn), len(w16)); a, c = xn[:m], w16[:m]
            p16 = float(pesq(SR, a, c, "wb")); sd16 = float(__import__("numpy").nan)
            from flowmse.util.other import si_sdr
            sd16 = float(si_sdr(a, c))
            mm = min(len(w32), len(w16))
            d_wav = np.abs(w32[:mm] - w16[:mm]); d_spec = np.abs(s32 - s16)
            collapsed = p16 < 2.0
            rec["autocast_first"] = {
                "pesq": p16, "si_sdr": sd16, "estoi": float(stoi(a, c, SR, extended=True)),
                "wav_max_abs_delta": float(d_wav.max()), "wav_rel_err_pct": float(d_wav.mean() / (np.abs(w32[:mm]).mean() + 1e-12) * 100),
                "spec_max_abs_delta": float(d_spec.max()), "spec_rel_err_pct": float(d_spec.mean() / (np.abs(s32).mean() + 1e-12) * 100),
                "half_dtype_buffers": dtypes,
                "collapsed": bool(collapsed)}
            n_collapsed += collapsed; n_clean += (not collapsed)
        except Exception as e:
            import traceback
            rec["autocast_first"] = {"crash": repr(e)[:200], "tb_last": traceback.format_exc().strip().splitlines()[-1][:200], "collapsed": True}
            n_collapsed += 1
        results.append(rec)
        r = rec["autocast_first"]
        print(f"  {bn}: pesq={r.get('pesq', r.get('crash','?'))} wav_rel={r.get('wav_rel_err_pct','-')}% "
              f"spec_rel={r.get('spec_rel_err_pct','-')}% dtypes={list(r.get('half_dtype_buffers',{}).items())[:2]}", flush=True)

    out = {"task": "fp16 autocast 独立跑(autocast 先行, 无 fp32 预跑) x10, pure 回退算子",
           "ops_env": {"NCSNPP_PURE_PYTORCH": os.environ.get("NCSNPP_PURE_PYTORCH"),
                        "NCSNPP_PURE_UPFIRDN": os.environ.get("NCSNPP_PURE_UPFIRDN")},
           "collapse_def": "PESQ < 2.0 (摧毁级=1.6-1.75, 正常≈2.89)",
           "k_of_10_ran_clean": n_clean, "k_of_10_collapsed": n_collapsed,
           "note": "不报数字入稿; §6.5 只引用定性结论(响亮失败 vs 静默通过)",
           "results": results}
    json.dump(out, open("a100_fp16_autocast_indep.json", "w"), indent=2)
    sha = hashlib.sha256(open("a100_fp16_autocast_indep.json", "rb").read()).hexdigest()
    open("a100_fp16_autocast_indep.json.sha256", "w").write(f"{sha}  a100_fp16_autocast_indep.json\n")
    print(f"\n✓ saved sha={sha[:16]} | ran_clean {n_clean}/10, collapsed {n_collapsed}/10", flush=True)

if __name__ == "__main__":
    main()
