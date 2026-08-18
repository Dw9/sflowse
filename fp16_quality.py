"""T2a 补充1: fp16 质量验证(autocast 混合精度) vs fp32 基准.
complex 谱模型无法纯 .half()(complex 无 fp16), 用 torch.autocast(fp16) 包 forward:
backbone 实数部分降 fp16, complex 边界(STFT/残差/iSTFT)保 fp32 = 混合精度.
质量未通过则 fp16 速度不报(导师补充1).
"""
import argparse, os, json, torch, numpy as np, soundfile as sf, torchaudio
from glob import glob
from pesq import pesq
from pystoi import stoi

SR = 16000


def si_sdr(x, xhat):
    x = x - np.mean(x); xhat = xhat - np.mean(xhat)
    alpha = np.dot(xhat, x) / (np.dot(x, x) + 1e-12)
    target = alpha * x
    noise = xhat - target
    return 10 * np.log10((np.sum(target ** 2) + 1e-12) / (np.sum(noise ** 2) + 1e-12))


SR = 16000


def si_sdr(x, xhat):
    x = x - np.mean(x); xhat = xhat - np.mean(xhat)
    a = np.sum(x * xhat) / (np.sum(x * xhat) ** 2 / np.sum(x * xhat) + 1e-12)  # placeholder
    # standard SI-SDR
    z = xhat - np.dot(xhat, x) / (np.dot(x, x) + 1e-12) * x
    a_opt = np.dot(x, x) / (np.dot(x, x) + 1e-12)
    target = a_opt * x
    noise = xhat - target
    return 10 * np.log10((np.sum(target ** 2) + 1e-12) / (np.sum(noise ** 2) + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="sflowse.ckpt")
    ap.add_argument("--data_dir", default="/home/zhibo/workspace/VoiceBank_processed")
    ap.add_argument("--num_files", type=int, default=824)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    from flowmse.util.other import pad_spec
    from flowmse.resflowse_model import ResFlowSEModel

    noisy = sorted(glob(os.path.join(args.data_dir, "test", "noisy", "*.wav")))
    clean = sorted(glob(os.path.join(args.data_dir, "test", "clean", "*.wav")))
    n = min(args.num_files, len(noisy), len(clean))
    noisy, clean = noisy[:n], clean[:n]

    model = ResFlowSEModel.load_from_checkpoint(args.ckpt, map_location="cpu",
                                               weights_only=False, strict=False)
    model.cuda().eval()

    pesqs, sdrs, estois = [], [], []
    n_nan = 0
    for i, (nf, cf) in enumerate(zip(noisy, clean)):
        y, sr = sf.read(nf); y = torch.from_numpy(y).float()
        if y.dim() == 1: y = y.unsqueeze(0)
        if sr != SR: y = torchaudio.functional.resample(y, sr, SR)
        x, _ = sf.read(cf); x = np.asarray(x).reshape(-1)
        T = y.size(1)
        norm = y.abs().max(); yn = y / norm
        Y = torch.unsqueeze(model._forward_transform(model._stft(yn.cuda())), 0)
        Y = pad_spec(Y)
        if args.fp16 or args.bf16:
            dt = torch.float16 if args.fp16 else torch.bfloat16
            with torch.no_grad(), torch.autocast("cuda", dtype=dt):
                xh = model.forward(Y)
            xh = xh.to(torch.complex64)
        else:
            with torch.no_grad():
                xh = model.forward(Y)
        xhat = model.to_audio(xh.squeeze(), T)
        xhat = (xhat * norm).squeeze().cpu().numpy().reshape(-1)
        if not np.isfinite(xhat).all():
            n_nan += 1; continue
        xhat = xhat[:len(x)] if len(xhat) >= len(x) else np.pad(xhat, (0, len(x) - len(xhat)))
        pesqs.append(pesq(SR, x, xhat, "wb"))
        estois.append(stoi(x, xhat, SR, extended=True))
        sdrs.append(float(si_sdr(x, xhat)))
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{n}] PESQ={np.mean(pesqs):.4f} nan_skipped={n_nan}", flush=True)

    prec = "fp16-autocast" if args.fp16 else ("bf16-autocast" if args.bf16 else "fp32")
    out = {"label": f"ResFlowSE (1-NFE) {prec}",
           "n_files": len(pesqs), "nan_skipped": n_nan,
           "metrics": {"pesq": {"mean": float(np.mean(pesqs)), "std": float(np.std(pesqs))},
                       "si_sdr": {"mean": float(np.mean(sdrs)), "std": float(np.std(sdrs))},
                       "estoi": {"mean": float(np.mean(estois)), "std": float(np.std(estois))}}}
    print(f"\n{out['label']}: PESQ {np.mean(pesqs):.6f} SI-SDR {np.mean(sdrs):.6f} "
          f"ESTOI {np.mean(estois):.6f} (n={len(pesqs)}, nan_skipped={n_nan})")
    if args.output:
        json.dump(out, open(args.output, "w"), indent=2); print("saved", args.output)


if __name__ == "__main__":
    main()
