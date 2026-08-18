"""a100_fp16_ten.py — fp16 十条完整路径证伪(REVIEW 2.1-2.4 版; SUBMIT_a100_dual PASS)
主分支=跑通+退化(ledger 已有 fp16 PESQ=1.752, 非崩); traceback 只是次要记录。
真正要拿的(2.2): fp16 vs fp32 **逐样本 max|delta| 与相对误差, 波形域+谱域各一** ——
complex32 实测误差仅 0.65% 不足以解释 PESQ 3.06→1.752, 该矛盾正是冻结表'主因未明'的原因。
10 条含 p232_001(2.4), 报 k/10 复现; 不设指标阈值判词(2.3); 不外推全数据集。"""
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


def enhance(model, y, T, half=False):
    yn = y / y.abs().max()
    Y = pad_spec(torch.unsqueeze(model._forward_transform(model._stft(yn.cuda())), 0))
    if half:
        Y = Y.half()
    with torch.no_grad():
        xh = model.forward(Y)
        wav = model.to_audio(xh.squeeze(), T)
    return wav.float().cpu().numpy(), (xh.squeeze()).float().cpu().numpy()


def spec_of(wav, model):
    t = torch.from_numpy(np.ascontiguousarray(wav)).float().cuda()
    S = model._stft(t.unsqueeze(0))
    return torch.view_as_complex(S).cpu().numpy()


def main():
    from flowmse.resflowse_model import ResFlowSEModel
    rfs = ResFlowSEModel.load_from_checkpoint("sflowse.ckpt", map_location="cpu", weights_only=False, strict=False)
    rfs.cuda().eval()

    clean_all = sorted(glob("/home/zhibo/workspace/VoiceBank_processed/test/clean/*.wav"))
    noisy_all = sorted(glob("/home/zhibo/workspace/VoiceBank_processed/test/noisy/*.wav"))
    # 10 条含 p232_001(首条) + 其余按时长散布
    idx = [0] + [int(x) for x in np.linspace(1, len(clean_all) - 1, 9)]
    files = [(clean_all[i], noisy_all[i]) for i in idx]
    print(f"[fp16 x10] files: {[os.path.basename(n) for _, n in files]}", flush=True)

    results = []
    ok_half = 0
    for cf, nf in files:
        bn = os.path.basename(nf)
        x = load_audio(cf); y = load_audio(nf); T = y.size(1)
        xn = x.squeeze().numpy()
        rec = {"file": bn}
        # fp32 基线
        try:
            w32, s32 = enhance(rfs, y, T, half=False)
            m = min(len(xn), len(w32)); a, b = xn[:m], w32[:m]
            rec["fp32"] = {"pesq": float(pesq(SR, a, b, "wb")), "si_sdr": float(si_sdr(a, b)),
                           "estoi": float(stoi(a, b, SR, extended=True))}
            sf.write(f"a100_fp16_{bn[:-4]}_fp32.wav", w32, SR, subtype="FLOAT")
        except Exception as e:
            rec["fp32"] = {"error": repr(e)[:200]}
        # fp16(.half())
        try:
            rfs_h = rfs.half()
            w16, s16 = enhance(rfs_h, y, T, half=True)
            rfs = rfs_h.float()  # 还原
            m = min(len(xn), len(w16)); a, c = xn[:m], w16[:m]
            mm = min(len(w32), len(w16))
            d_wav = np.abs(w32[:mm] - w16[:mm])
            d_spec = np.abs(s32 - s16)
            rec["fp16_half"] = {
                "pesq": float(pesq(SR, a, c, "wb")), "si_sdr": float(si_sdr(a, c)),
                "estoi": float(stoi(a, c, SR, extended=True)),
                "wav_max_abs_delta": float(d_wav.max()),
                "wav_rel_err_pct": float(d_wav.mean() / (np.abs(w32[:mm]).mean() + 1e-12) * 100),
                "spec_max_abs_delta": float(d_spec.max()),
                "spec_rel_err_pct": float(d_spec.mean() / (np.abs(s32).mean() + 1e-12) * 100),
            }
            sf.write(f"a100_fp16_{bn[:-4]}_fp16.wav", w16, SR, subtype="FLOAT")
            ok_half += 1
        except Exception as e:
            import traceback
            rec["fp16_half"] = {"crash": repr(e)[:200], "tb_first": traceback.format_exc().strip().splitlines()[-1][:200]}
            try: rfs = rfs.float()
            except Exception: pass
        results.append(rec)
        print(f"  {bn}: fp32={rec.get('fp32',{}).get('pesq','-')} fp16={rec.get('fp16_half',{}).get('pesq', rec.get('fp16_half',{}).get('crash','?'))}", flush=True)

    out = {"task": "fp16 十条完整路径证伪(REVIEW 2.1-2.4)",
           "n_files": len(files), "k_of_10_reproduced_half": ok_half,
           "expected_branch": "跑通+退化(ledger fp16 PESQ 1.752 非崩); traceback 次要",
           "target_2_2": "逐样本 max|delta| 波形+谱域(complex32 0.65% 不足以解释 PESQ 塌, 矛盾=冻结表'主因未明'的原因)",
           "no_threshold_2_3": "不设指标阈值判词, 直接报实测",
           "no_extrapolation": "10/10 或 k/10 复现; 不外推全数据集",
           "results": results}
    json.dump(out, open("a100_fp16_ten.json", "w"), indent=2)
    sha = hashlib.sha256(open("a100_fp16_ten.json", "rb").read()).hexdigest()
    open("a100_fp16_ten.json.sha256", "w").write(f"{sha}  a100_fp16_ten.json\n")
    print(f"\n✓ saved  sha={sha[:16]}  | half 通过 {ok_half}/10", flush=True)


if __name__ == "__main__":
    main()
