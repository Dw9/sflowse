"""
Experiment B: per-band complex & magnitude spectral MSE (vs clean) + identity check.

For three configs over the FULL VoiceBank-DEMAND test set (824 files):
  - FlowSE truncated N=1   (generative, expected lowest spectral MSE / over-smoothed)
  - FlowSE teacher   N=5   (generative)
  - Proposed (ResFlowSE)   (discriminative)

Outputs:
  1. Identity table: PESQ / SI-SDR / ESTOI  (must reproduce known 2.889 / 3.089 / 3.062)
  2. Per-band complex spectral MSE  (low 0-1k / mid 1-4k / high 4-8k), mean over 824
  3. Per-band magnitude spectral MSE (same bands)

STFT uses the model's exact params: n_fft=510, hop=128, hann(periodic=True), center=True.
Enhanced wavs are produced with the SAME normalization pipeline as eval_metrics.py, so the
enhanced (denormalized) and clean (raw) waveforms are on a comparable absolute scale.
"""
import argparse, os, json
import numpy as np
import torch
import torch.serialization
import torchaudio
import soundfile as sf
from glob import glob
from tqdm import tqdm
from pesq import pesq
from pystoi import stoi

from flowmse.resflowse_model import ResFlowSEModel
from flowmse.data_module import SpecsDataModule
from flowmse.model import VFModel
from flowmse.sampling import get_white_box_solver
from flowmse.util.other import si_sdr, pad_spec
from eval_metrics import load_audio_16k, enhance_resflowse, enhance_flowse

try:
    torch.serialization.add_safe_globals([SpecsDataModule])
except AttributeError:
    pass

SR = 16000
N_FFT, HOP, WIN = 510, 128, "hann"
# bin frequency: k * SR / N_FFT. 1000Hz -> 31.9, 4000Hz -> 127.5  (256 bins total)
BANDS = {"low (0-1k)":  (0, 32), "mid (1-4k)": (32, 128), "high (4-8k)": (128, 256)}


def stft_complex(wav_np, device, window):
    """wav_np: 1D float32. Returns complex STFT [F=256, T] on device."""
    t = torch.from_numpy(np.ascontiguousarray(wav_np)).float().to(device)
    S = torch.stft(t, n_fft=N_FFT, hop_length=HOP, window=window.to(device),
                   center=True, return_complex=True)
    return S  # [F, T] complex


def band_mse(S_enh, S_clean):
    """Returns dict: per-band complex MSE and magnitude MSE (mean over band x time)."""
    out = {}
    d = S_enh - S_clean                       # complex difference
    cplx_sq = (d.real ** 2 + d.imag ** 2)     # |d|^2  [F, T]
    mag = (S_enh.abs() - S_clean.abs()) ** 2  # (|enh|-|cln|)^2 [F,T]
    for name, (lo, hi) in BANDS.items():
        out[name + " cplx"]  = float(cplx_sq[lo:hi].mean().item())
        out[name + " mag"]   = float(mag[lo:hi].mean().item())
    out["full cplx"] = float(cplx_sq.mean().item())
    out["full mag"]  = float(mag.mean().item())
    return out


def run_config(label, model_kind, enhance_fn, clean_files, noisy_files, device, window, num_files=None):
    if num_files:
        idx = np.linspace(0, len(clean_files) - 1, num_files, dtype=int)
        clean_files = [clean_files[i] for i in idx]
        noisy_files = [noisy_files[i] for i in idx]
    n = len(clean_files)

    # identity accumulators
    pesq_l, sisd_l, estoi_l = [], [], []
    # band accumulators
    band_keys = [b + s for b in BANDS for s in (" cplx", " mag")] + ["full cplx", "full mag"]
    band_sum = {k: 0.0 for k in band_keys}

    for cf, nf in tqdm(zip(clean_files, noisy_files), total=n, desc=label):
        x = load_audio_16k(cf)
        y = load_audio_16k(nf)
        T_orig = x.size(1)
        x_np = x.squeeze().numpy()
        x_hat_np = enhance_fn(y, T_orig)
        m = min(len(x_np), len(x_hat_np))
        x_np, x_hat_np = x_np[:m], x_hat_np[:m]

        # identity metrics
        pesq_l.append(pesq(SR, x_np, x_hat_np, "wb"))
        sisd_l.append(si_sdr(x_np, x_hat_np))
        estoi_l.append(stoi(x_np, x_hat_np, SR, extended=True))

        # spectral band MSE (vs clean)
        S_enh = stft_complex(x_hat_np, device, window)
        S_cln = stft_complex(x_np, device, window)
        T = min(S_enh.shape[1], S_cln.shape[1])
        bm = band_mse(S_enh[:, :T], S_cln[:, :T])
        for k in band_keys:
            band_sum[k] += bm[k]

    res = {
        "label": label, "n": n, "model_kind": model_kind,
        "pesq": float(np.mean(pesq_l)), "si_sdr": float(np.mean(sisd_l)),
        "estoi": float(np.mean(estoi_l)),
        "band": {k: band_sum[k] / n for k in band_keys},
    }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/home/zhibo/workspace/VoiceBank_processed")
    ap.add_argument("--split", default="test")
    ap.add_argument("--flowse_ckpt", default="VB_DMD_FLOWSE_ICASSP_2025.ckpt")
    ap.add_argument("--resflowse_ckpt", default="sflowse.ckpt")
    ap.add_argument("--no_ema", action="store_true")
    ap.add_argument("--num_files", type=int, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="band_mse_results.json")
    args = ap.parse_args()

    device = torch.device(args.device)
    window = torch.hann_window(N_FFT, periodic=True)

    clean_files = sorted(glob(os.path.join(args.data_dir, args.split, "clean", "*.wav")))
    noisy_files = sorted(glob(os.path.join(args.data_dir, args.split, "noisy", "*.wav")))
    assert len(clean_files) == len(noisy_files) and len(clean_files) > 0
    print(f"Found {len(clean_files)} files in {args.split}")

    # ---- FlowSE teacher (loads once, used for N=1 and N=5) ----
    print(f"\nLoading FlowSE: {args.flowse_ckpt}")
    flowse = VFModel.load_from_checkpoint(args.flowse_ckpt, base_dir=args.data_dir,
                                          map_location="cpu")
    for name, param in flowse.dnn.named_parameters():
        if name in flowse.ema_dnn:
            param.data = flowse.ema_dnn[name].to(param.device)
    flowse.eval(); flowse.to(device)

    results = []
    for N, lbl in [(1, "FlowSE (N=1, truncated)"), (5, "FlowSE (N=5, teacher)")]:
        fn = lambda y, T, _N=N: enhance_flowse(flowse, y, T, N=_N)
        results.append(run_config(lbl, "flowse", fn, clean_files, noisy_files,
                                  device, window, args.num_files))
        print(f"  [{lbl}] PESQ={results[-1]['pesq']:.3f} SI-SDR={results[-1]['si_sdr']:.2f} "
              f"ESTOI={results[-1]['estoi']:.3f}")
        # free GPU cache between configs
        torch.cuda.empty_cache()

    # ---- Proposed (ResFlowSE) ----
    print(f"\nLoading ResFlowSE: {args.resflowse_ckpt}")
    rf = ResFlowSEModel.load_from_checkpoint(args.resflowse_ckpt, map_location="cpu",
                                             weights_only=False, strict=False)
    rf.eval()
    if not args.no_ema:
        rf._swap_to_ema()
    rf.to(device)
    fn = lambda y, T: enhance_resflowse(rf, y, T)
    results.append(run_config("Proposed (M3)", "resflowse", fn, clean_files, noisy_files,
                              device, window, args.num_files))
    print(f"  [Proposed (M3)] PESQ={results[-1]['pesq']:.3f} SI-SDR={results[-1]['si_sdr']:.2f} "
          f"ESTOI={results[-1]['estoi']:.3f}")

    # ===================== REPORT =====================
    print("\n" + "=" * 78)
    print("IDENTITY CHECK (must match known: N=1=2.889/19.54 | N=5=3.089/18.85 | M3=3.062/18.79)")
    print("=" * 78)
    print(f"{'Config':<28}{'PESQ':>8}{'SI-SDR':>9}{'ESTOI':>8}")
    for r in results:
        print(f"{r['label']:<28}{r['pesq']:>8.3f}{r['si_sdr']:>9.2f}{r['estoi']:>8.3f}")

    print("\n" + "=" * 78)
    print("PER-BAND COMPLEX SPECTRAL MSE  (|S_enh - S_clean|^2, mean over 824)  -- LOWER = smoother")
    print("=" * 78)
    hdr = f"{'Config':<28}" + "".join(f"{b:>16}" for b in BANDS) + f"{'full':>14}"
    print(hdr)
    for r in results:
        b = r["band"]
        row = f"{r['label']:<28}"
        for bn in BANDS:
            row += f"{b[bn+' cplx']:>16.5g}"
        row += f"{b['full cplx']:>14.5g}"
        print(row)

    print("\n" + "=" * 78)
    print("PER-BAND MAGNITUDE SPECTRAL MSE  ((|S_enh|-|S_clean|)^2, mean over 824)")
    print("=" * 78)
    print(hdr)
    for r in results:
        b = r["band"]
        row = f"{r['label']:<28}"
        for bn in BANDS:
            row += f"{b[bn+' mag']:>16.5g}"
        row += f"{b['full mag']:>14.5g}"
        print(row)

    with open(args.out, "w") as f:
        json.dump({"n_files": results[0]["n"], "configs": results}, f, indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
