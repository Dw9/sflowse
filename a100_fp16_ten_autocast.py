"""a100_fp16_ten_autocast.py — fp16 autocast 分支补跑(导师指认 1.752 路径; 取 2.2 逐样本 max|delta|)
源路径: eval_t2a_fp16.json = 'ResFlowSE (1-NFE) fp16-autocast', n=824, PESQ 1.7521 / SI-SDR -1.2706 / ESTOI -0.0011
(信号被摧毁级, 非"精度退化"; 0.65% complex32 数值误差不足以解释 —— 若 max|delta| 也仅百分之几,
即又一例"机制成立但不足够", 照实报不圆)。同 10 条(含 p232_001), 不设阈值不外推。"""
import os, json, hashlib
import torch, numpy as np, soundfile as sf, torchaudio
os.environ['PATH'] = os.path.expanduser('~/.local/bin') + ':/usr/local/cuda/bin:' + os.environ['PATH']
from glob import glob
from pesq import pesq
from pystoi import stoi
from flowmse.util.other import si_sdr, pad_spec
SR = 16000


def load_audio(fp, sr=SR):
    y, sr0 = sf.read(fp); y = torch.from_numpy(y).float()
    if y.dim() == 1: y = y.unsqueeze(0)
    if sr0 != sr: y = torchaudio.functional.resample(y, sr0, sr)
    return y


def enhance(model, y, T, autocast=False):
    yn = y / y.abs().max()
    Y = pad_spec(torch.unsqueeze(model._forward_transform(model._stft(yn.cuda())), 0))
    with torch.no_grad():
        if autocast:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                xh = model.forward(Y)
            xh = xh.to(torch.complex64)   # 关键: 原路径 fp16_quality.py L73, 漏它必崩 'Half vs Float'
        else:
            with torch.no_grad():
                xh = model.forward(Y)
        wav = model.to_audio(xh.squeeze(), T)
    return wav.float().cpu().numpy(), xh.squeeze().float().cpu().numpy()


def main():
    from flowmse.resflowse_model import ResFlowSEModel
    rfs = ResFlowSEModel.load_from_checkpoint("sflowse.ckpt", map_location="cpu", weights_only=False, strict=False)
    rfs.cuda().eval()
    clean_all = sorted(glob("/home/zhibo/workspace/VoiceBank_processed/test/clean/*.wav"))
    noisy_all = sorted(glob("/home/zhibo/workspace/VoiceBank_processed/test/noisy/*.wav"))
    idx = [0] + [int(x) for x in np.linspace(1, len(clean_all) - 1, 9)]
    files = [(clean_all[i], noisy_all[i]) for i in idx]
    print(f"[fp16 autocast x10] {[os.path.basename(n) for _, n in files]}", flush=True)

    results, ok = [], 0
    for cf, nf in files:
        bn = os.path.basename(nf); rec = {"file": bn}
        x = load_audio(cf); y = load_audio(nf); T = y.size(1); xn = x.squeeze().numpy()
        try:
            w32, s32 = enhance(rfs, y, T, autocast=False)
            m = min(len(xn), len(w32)); a, b = xn[:m], w32[:m]
            rec["fp32"] = {"pesq": float(pesq(SR, a, b, "wb")), "si_sdr": float(si_sdr(a, b)), "estoi": float(stoi(a, b, SR, extended=True))}
        except Exception as e:
            rec["fp32"] = {"error": repr(e)[:200]}
        try:
            w16, s16 = enhance(rfs, y, T, autocast=True)
            m = min(len(xn), len(w16)); a, c = xn[:m], w16[:m]
            mm = min(len(w32), len(w16))
            d_wav = np.abs(w32[:mm] - w16[:mm]); d_spec = np.abs(s32 - s16)
            rec["fp16_autocast"] = {
                "pesq": float(pesq(SR, a, c, "wb")), "si_sdr": float(si_sdr(a, c)), "estoi": float(stoi(a, c, SR, extended=True)),
                "wav_max_abs_delta": float(d_wav.max()), "wav_rel_err_pct": float(d_wav.mean() / (np.abs(w32[:mm]).mean() + 1e-12) * 100),
                "spec_max_abs_delta": float(d_spec.max()), "spec_rel_err_pct": float(d_spec.mean() / (np.abs(s32).mean() + 1e-12) * 100),
                "wav_max_rel_pct_at_peak": float(d_wav.max() / (np.abs(w32[:mm]).max() + 1e-12) * 100),
            }
            sf.write(f"a100_fp16_{bn[:-4]}_autocast.wav", w16, SR, subtype="FLOAT")
            ok += 1
        except Exception as e:
            import traceback
            rec["fp16_autocast"] = {"crash": repr(e)[:200], "tb_first": traceback.format_exc().strip().splitlines()[-1][:200]}
        results.append(rec)
        r = rec.get("fp16_autocast", {})
        print(f"  {bn}: fp32={rec['fp32'].get('pesq','-')} auto: pesq={r.get('pesq', r.get('crash','?'))} wav_rel={r.get('wav_rel_err_pct','-')}% spec_rel={r.get('spec_rel_err_pct','-')}%", flush=True)

    out = {"task": "fp16 autocast 十条(导师指认 1.752 路径; 2.2 max|delta| 目标)",
           "provenance": "eval_t2a_fp16.json: 'ResFlowSE (1-NFE) fp16-autocast' n=824 PESQ 1.7521/SI-SDR -1.2706/ESTOI -0.0011(信号摧毁级)",
           "reading_guide": "0.65% complex32 不足以解释摧毁; 若 max|delta| 也仅百分之几 = 机制成立但不足够, 照实报不圆",
           "n_files": len(files), "k_of_10_reproduced_autocast": ok, "results": results}
    json.dump(out, open("a100_fp16_ten_autocast.json", "w"), indent=2)
    sha = hashlib.sha256(open("a100_fp16_ten_autocast.json", "rb").read()).hexdigest()
    open("a100_fp16_ten_autocast.json.sha256", "w").write(f"{sha}  a100_fp16_ten_autocast.json\n")
    print(f"\n✓ saved sha={sha[:16]} | autocast 通过 {ok}/10", flush=True)


if __name__ == "__main__":
    main()
