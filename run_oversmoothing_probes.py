"""
Experiment B supplementary probes (mentor directive): does N=1 really over-smooth?

Rationale: complex spectral L2  |S_enh-S_clean|^2  is, by Parseval, ~= time-domain L2 ~= the
spectral form of SI-SDR. So "N=1 lowest complex L2" and "N=1 highest SI-SDR" are REDUNDANT, not
independent. Over-smoothing must be shown in the MAGNITUDE / PERCEPTUAL domain. Three probes:

  (1) Magnitude 3-band MSE (low/mid/high) -- is it monotonic N=1<N=5<M3 ?
  (2) High-freq (4-8k) energy retention = (HF_energy/full_energy)_enh / (...)_clean  (per file)
      over-smoothing => enhanced loses HF harmonic detail => retention < 1 and N=1 lowest.
  (3) Spectral flatness (librosa.feature.spectral_flatness) mean -- flatter => over-smoothed.

M3 = sflowse.ckpt loaded into current resflowse_model.py with NO EMA (reproduces PESQ 3.062).
N=1 / N=5 = VB_DMD_FLOWSE teacher (EMA applied), reproduce 2.893 / 3.089.
"""
import os, json
import numpy as np
import torch
import librosa
from glob import glob
from tqdm import tqdm

from flowmse.resflowse_model import ResFlowSEModel
from flowmse.model import VFModel
from eval_metrics import load_audio_16k, enhance_resflowse, enhance_flowse

SR = 16000
N_FFT, HOP = 510, 128
WIN = torch.hann_window(N_FFT, periodic=True)
BANDS = {"low(0-1k)": (0, 32), "mid(1-4k)": (32, 128), "high(4-8k)": (128, 256)}
HI = (128, 256)  # high-freq energy band
DEVICE = torch.device("cuda:0")


def stft(wav_np):
    t = torch.from_numpy(np.ascontiguousarray(wav_np)).float().to(DEVICE)
    return torch.stft(t, n_fft=N_FFT, hop_length=HOP, window=WIN.to(DEVICE),
                      center=True, return_complex=True)  # [F,T]


def probes(x_hat_np, x_np):
    S = stft(x_hat_np)                       # enhanced complex STFT
    Sc = stft(x_np)                          # clean complex STFT
    T = min(S.shape[1], Sc.shape[1])
    S, Sc = S[:, :T], Sc[:, :T]
    mag, magc = S.abs(), Sc.abs()

    # (1) magnitude 3-band MSE
    md = ((mag - magc) ** 2)
    mag_mse = {b: float(md[lo:hi].mean().item()) for b, (lo, hi) in BANDS.items()}
    mag_mse["full"] = float(md.mean().item())

    # (2) high-freq energy retention (per file): HF_ratio_enh / HF_ratio_clean
    pow = mag ** 2
    powc = magc ** 2
    hf_enh = float(pow[HI[0]:HI[1]].sum().item()); full_enh = float(pow.sum().item())
    hf_cln = float(powc[HI[0]:HI[1]].sum().item()); full_cln = float(powc.sum().item())
    hf_ratio_enh = hf_enh / (full_enh + 1e-20)
    hf_ratio_cln = hf_cln / (full_cln + 1e-20)
    retention = hf_ratio_enh / (hf_ratio_cln + 1e-20)

    # (3) spectral flatness (librosa, default n_fft=2048, hop=512) on enhanced & clean
    sf_enh = float(np.mean(librosa.feature.spectral_flatness(y=x_hat_np.astype(np.float64))))
    sf_cln = float(np.mean(librosa.feature.spectral_flatness(y=x_np.astype(np.float64))))

    return {
        "mag_mse": mag_mse,
        "hf_ratio_enh": hf_ratio_enh, "hf_ratio_cln": hf_ratio_cln, "hf_retention": retention,
        "sf_enh": sf_enh, "sf_cln": sf_cln,
    }


def new_acc():
    return {  # per-config accumulators
        "n": 0,
        "mag_mse": {b: 0.0 for b in list(BANDS) + ["full"]},
        "hf_ratio_enh": 0.0, "hf_ratio_cln": 0.0, "hf_retention": 0.0,
        "sf_enh": 0.0, "sf_cln": 0.0,
    }


def add(acc, p):
    acc["n"] += 1
    for b in acc["mag_mse"]:
        acc["mag_mse"][b] += p["mag_mse"][b]
    acc["hf_ratio_enh"] += p["hf_ratio_enh"]
    acc["hf_ratio_cln"] += p["hf_ratio_cln"]
    acc["hf_retention"] += p["hf_retention"]
    acc["sf_enh"] += p["sf_enh"]
    acc["sf_cln"] += p["sf_cln"]


def main():
    data_dir = "/home/zhibo/workspace/VoiceBank_processed"
    cf = sorted(glob(os.path.join(data_dir, "test", "clean", "*.wav")))
    nf = sorted(glob(os.path.join(data_dir, "test", "noisy", "*.wav")))
    assert len(cf) == len(nf) and len(cf) > 0
    N = len(cf)
    print(f"files={N}")

    # FlowSE teacher (N=1 and N=5)
    flowse = VFModel.load_from_checkpoint("VB_DMD_FLOWSE_ICASSP_2025.ckpt", base_dir=data_dir,
                                          map_location="cpu")
    for name, param in flowse.dnn.named_parameters():
        if name in flowse.ema_dnn:
            param.data = flowse.ema_dnn[name].to(param.device)
    flowse.eval(); flowse.to(DEVICE)

    # Proposed M3 = sflowse.ckpt NO EMA
    rf = ResFlowSEModel.load_from_checkpoint("sflowse.ckpt", map_location="cpu",
                                             weights_only=False, strict=False)
    rf.eval()  # NO EMA swap
    rf.to(DEVICE)

    acc = {"N=1": new_acc(), "N=5": new_acc(), "M3": new_acc()}
    for c, ny in tqdm(zip(cf, nf), total=N):
        x = load_audio_16k(c); y = load_audio_16k(ny); T = x.size(1)
        x_np = x.squeeze().numpy()
        for key, fn in (("N=1", lambda yy, Tt: enhance_flowse(flowse, yy, Tt, N=1)),
                        ("N=5", lambda yy, Tt: enhance_flowse(flowse, yy, Tt, N=5)),
                        ("M3", lambda yy, Tt: enhance_resflowse(rf, yy, Tt))):
            hn = fn(y, T)
            m = min(len(x_np), len(hn))
            add(acc[key], probes(hn[:m], x_np[:m]))
        if acc["N=1"]["n"] % 100 == 0:
            print(f"  [{acc['N=1']['n']}]")

    # finalize means
    for k in acc:
        n = acc[k]["n"]
        for b in acc[k]["mag_mse"]:
            acc[k]["mag_mse"][b] /= n
        for f in ("hf_ratio_enh", "hf_ratio_cln", "hf_retention", "sf_enh", "sf_cln"):
            acc[k][f] /= n

    # ================= REPORT =================
    keys = ["N=1", "N=5", "M3"]
    print("\n" + "=" * 86)
    print("(1) MAGNITUDE spectral MSE per band  ((|enh|-|cln|)^2, mean 824)  -- lower=less mag error")
    print("=" * 86)
    print(f"{'cfg':<6}" + "".join(f"{b:>16}" for b in list(BANDS) + ["full"]))
    for k in keys:
        row = f"{k:<6}"
        for b in list(BANDS) + ["full"]:
            row += f"{acc[k]['mag_mse'][b]:>16.6g}"
        print(row)
    mono = all(acc["N=1"]["mag_mse"][b] < acc["N=5"]["mag_mse"][b] < acc["M3"]["mag_mse"][b]
               for b in list(BANDS) + ["full"])
    print(f"  -> monotonic N=1<N=5<M3 in ALL bands? {mono}")

    print("\n" + "=" * 86)
    print("(2) HIGH-FREQ (4-8k) ENERGY RETENTION = (HF/full)_enh / (HF/full)_clean  (<1 => lost HF)")
    print("=" * 86)
    print(f"{'cfg':<6}{'HF_ratio_enh':>16}{'HF_ratio_clean':>18}{'retention=enh/cln':>20}")
    for k in keys:
        print(f"{k:<6}{acc[k]['hf_ratio_enh']:>16.5f}{acc[k]['hf_ratio_cln']:>18.5f}{acc[k]['hf_retention']:>20.4f}")
    print(f"  -> N=1 retention lowest (=most HF loss)? "
          f"{acc['N=1']['hf_retention'] < acc['N=5']['hf_retention'] < acc['M3']['hf_retention']}")

    print("\n" + "=" * 86)
    print("(3) SPECTRAL FLATNESS (librosa, mean 824)  -- HIGHER = flatter/blunter = over-smoothed")
    print("=" * 86)
    print(f"{'cfg':<6}{'flatness_enh':>16}{'flatness_clean':>20}")
    for k in keys:
        print(f"{k:<6}{acc[k]['sf_enh']:>16.5f}{acc[k]['sf_cln']:>20.5f}")
    print(f"  -> N=1 flatter than N=5 & M3? "
          f"{acc['N=1']['sf_enh'] > acc['N=5']['sf_enh'] and acc['N=1']['sf_enh'] > acc['M3']['sf_enh']}")

    out = {k: acc[k] for k in keys}
    out["_meta"] = {"n_files": N, "hf_band_bins": list(HI), "bands": BANDS,
                    "note": "M3=sflowse.ckpt no-EMA(=3.062); N=1/N=5=VB teacher(EMA)"}
    json.dump(out, open("band_mse_probes.json", "w"), indent=2)
    print("\nsaved band_mse_probes.json")


if __name__ == "__main__":
    main()
