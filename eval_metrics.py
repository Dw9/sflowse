"""
Unified evaluation script: FlowSE vs ResFlowSE
Computes PESQ, SI-SDR, ESTOI, DNSMOS (SIG/BAK/OVRL/P808) for both models.

Usage:
  python eval_metrics.py \
      --resflowse_ckpt logs/resflowse_.../epoch=39_pesq=2.94.ckpt \
      --flowse_ckpt VB_DMD_FLOWSE_ICASSP_2025.ckpt \
      --data_dir /home/zhibo/workspace/VoiceBank_processed

Requirements (install if missing):
  pip install onnxruntime
  git clone https://github.com/microsoft/DNS-Challenge.git  (for DNSMOS .onnx models)
  set --dnsmos_dir to the cloned DNS-Challenge/DNSMOS folder
"""

import argparse
import os
import sys
import tempfile
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
from flowmse.data_module import SpecsDataModule, load_audio as flowse_load_audio
from flowmse.model import VFModel
from flowmse.sampling import get_white_box_solver
from flowmse.util.other import si_sdr, pad_spec

torch.serialization.add_safe_globals([SpecsDataModule])

SR = 16000


# ---------------------------------------------------------------------------
# DNSMOS
# ---------------------------------------------------------------------------

def load_dnsmos(dnsmos_dir):
    """Load DNSMOS ONNX models. Returns (sig_bak_ovr_session, p808_session)."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed. Run: pip install onnxruntime")
        return None, None

    sig_bak_path = os.path.join(dnsmos_dir, "sig_bak_ovr.onnx")
    p808_path    = os.path.join(dnsmos_dir, "model_v8.onnx")

    if not os.path.exists(sig_bak_path) or not os.path.exists(p808_path):
        print(f"DNSMOS .onnx files not found in {dnsmos_dir}. Skipping DNSMOS.")
        return None, None

    sess_sbo = ort.InferenceSession(sig_bak_path, providers=["CPUExecutionProvider"])
    sess_p808 = ort.InferenceSession(p808_path,   providers=["CPUExecutionProvider"])
    return sess_sbo, sess_p808


def _audio_melspec(audio, n_mels=120, frame_size=320, hop_length=160, sr=16000):
    import librosa
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=frame_size+1, hop_length=hop_length, n_mels=n_mels
    )
    mel_spec = (librosa.power_to_db(mel_spec, ref=np.max) + 40) / 40
    return mel_spec.T  # (T, n_mels)


def _polyfit_val(sig, bak, ovr):
    import numpy.polynomial.polynomial as poly
    p_ovr = np.poly1d([-0.06766283,  1.11546468,  0.04602535])
    p_sig = np.poly1d([-0.08397278,  1.22083953,  0.0052439 ])
    p_bak = np.poly1d([-0.13166888,  1.60915514, -0.39604546])
    return float(p_sig(sig)), float(p_bak(bak)), float(p_ovr(ovr))


def compute_dnsmos_single(wav_np, sess_sbo, sess_p808, sr=16000):
    """Compute DNSMOS for a single waveform (numpy float32 array).
    Follows official dnsmos_local.py: sliding 9.01s window, polyfit post-processing.
    """
    INPUT_LENGTH = 9.01
    len_samples = int(INPUT_LENGTH * sr)

    # Repeat-pad if shorter than one window
    audio = wav_np.copy()
    while len(audio) < len_samples:
        audio = np.append(audio, audio)

    num_hops = int(np.floor(len(audio) / sr) - INPUT_LENGTH) + 1
    hop_len_samples = sr

    sigs, baks, ovrs, p808s = [], [], [], []
    for idx in range(num_hops):
        seg = audio[int(idx * hop_len_samples): int((idx + INPUT_LENGTH) * hop_len_samples)]
        if len(seg) < len_samples:
            continue

        wav_in  = np.array(seg).astype("float32")[np.newaxis, :]
        mel_in  = np.array(_audio_melspec(seg[:-160])).astype("float32")[np.newaxis, :, :]

        p808_mos = sess_p808.run(None, {"input_1": mel_in})[0][0][0]
        sig_raw, bak_raw, ovr_raw = sess_sbo.run(None, {"input_1": wav_in})[0][0]
        sig, bak, ovr = _polyfit_val(sig_raw, bak_raw, ovr_raw)

        sigs.append(sig); baks.append(bak); ovrs.append(ovr); p808s.append(p808_mos)

    return float(np.mean(sigs)), float(np.mean(baks)), float(np.mean(ovrs)), float(np.mean(p808s))


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def load_audio_16k(filepath):
    waveform, sr = sf.read(filepath)
    waveform = torch.from_numpy(waveform).float()
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if sr != SR:
        waveform = torchaudio.functional.resample(waveform, sr, SR)
    return waveform


# ---------------------------------------------------------------------------
# Model inference
# ---------------------------------------------------------------------------

def enhance_resflowse(model, y, T_orig):
    norm_factor = y.abs().max()
    y_norm = y / norm_factor
    Y = torch.unsqueeze(model._forward_transform(model._stft(y_norm.cuda())), 0)
    Y = pad_spec(Y)
    with torch.no_grad():
        x_hat_spec = model.forward(Y)
    x_hat = model.to_audio(x_hat_spec.squeeze(), T_orig)
    return (x_hat * norm_factor).squeeze().cpu().numpy()


def enhance_flowse(model, y, T_orig, N=5):
    norm_factor = y.abs().max()
    y_norm = y / norm_factor
    Y = torch.unsqueeze(model._forward_transform(model._stft(y_norm.cuda())), 0)
    Y = pad_spec(Y)
    with torch.no_grad():
        sampler = get_white_box_solver(
            "euler", model.ode, model, Y.cuda(),
            T_rev=model.T_rev, t_eps=model.t_eps, N=N
        )
        sample, _ = sampler()
    x_hat = model.to_audio(sample.squeeze(), T_orig)
    return (x_hat * norm_factor).squeeze().cpu().numpy()


# ---------------------------------------------------------------------------
# Per-file metrics
# ---------------------------------------------------------------------------

def compute_intrusive(x_np, x_hat_np):
    p = pesq(SR, x_np, x_hat_np, "wb")
    s = si_sdr(x_np, x_hat_np)
    e = stoi(x_np, x_hat_np, SR, extended=True)
    return p, s, e


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate_model(name, enhance_fn, clean_files, noisy_files, sess_sbo, sess_p808):
    results = {"filename": [], "pesq": [], "si_sdr": [], "estoi": [],
               "dnsmos_sig": [], "dnsmos_bak": [], "dnsmos_ovrl": [], "p808": []}

    for i, (cf, nf) in enumerate(tqdm(zip(clean_files, noisy_files),
                                       total=len(clean_files), desc=name)):
        x = load_audio_16k(cf)
        y = load_audio_16k(nf)
        T_orig = x.size(1)

        x_np = x.squeeze().numpy()
        x_hat_np = enhance_fn(y, T_orig)

        # Align length
        min_len = min(len(x_np), len(x_hat_np))
        x_np = x_np[:min_len]
        x_hat_np = x_hat_np[:min_len]

        p, s, e = compute_intrusive(x_np, x_hat_np)
        results["filename"].append(os.path.basename(cf))
        results["pesq"].append(p)
        results["si_sdr"].append(s)
        results["estoi"].append(e)

        if sess_sbo is not None:
            sig, bak, ovrl, p808 = compute_dnsmos_single(x_hat_np, sess_sbo, sess_p808)
            results["dnsmos_sig"].append(sig)
            results["dnsmos_bak"].append(bak)
            results["dnsmos_ovrl"].append(ovrl)
            results["p808"].append(p808)

        if (i + 1) % 100 == 0:
            tqdm.write(f"  [{i+1}] PESQ={p:.3f} SI-SDR={s:.2f} ESTOI={e:.3f}")

    return results


def print_results(name, results):
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    metrics = ["pesq", "si_sdr", "estoi"]
    if results["dnsmos_sig"]:
        metrics += ["dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl", "p808"]
    for k in metrics:
        v = np.array(results[k])
        print(f"  {k.upper():>12s}: {np.mean(v):.4f} ± {np.std(v):.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type",    type=str, required=True,
                        choices=["resflowse", "flowse"],
                        help="Which model to evaluate")
    parser.add_argument("--ckpt",          type=str, required=True,
                        help="Path to checkpoint")
    parser.add_argument("--data_dir",      type=str,
                        default="/home/zhibo/workspace/VoiceBank_processed")
    parser.add_argument("--split",         type=str, default="test")
    parser.add_argument("--N",             type=int, default=5,
                        help="FlowSE ODE steps (ignored for ResFlowSE)")
    parser.add_argument("--no_ema",        action="store_true",
                        help="ResFlowSE: skip EMA weight swap")
    parser.add_argument("--dnsmos_dir",    type=str, default=None,
                        help="Path to DNS-Challenge/DNSMOS folder with .onnx files")
    parser.add_argument("--num_files",     type=int, default=None,
                        help="Limit number of files (for quick testing)")
    parser.add_argument("--output",        type=str, default=None,
                        help="Save results to this txt file")
    parser.add_argument("--output_csv",     type=str, default=None,
                        help="Save per-utterance results to this CSV file")
    args = parser.parse_args()

    # DNSMOS models
    sess_sbo, sess_p808 = None, None
    if args.dnsmos_dir:
        sess_sbo, sess_p808 = load_dnsmos(args.dnsmos_dir)

    # File lists
    clean_dir = os.path.join(args.data_dir, args.split, "clean")
    noisy_dir = os.path.join(args.data_dir, args.split, "noisy")
    clean_files = sorted(glob(os.path.join(clean_dir, "*.wav")))
    noisy_files = sorted(glob(os.path.join(noisy_dir, "*.wav")))
    assert len(clean_files) == len(noisy_files)
    print(f"Found {len(clean_files)} files in {args.split} split")

    if args.num_files:
        idx = np.linspace(0, len(clean_files)-1, args.num_files, dtype=int)
        clean_files = [clean_files[i] for i in idx]
        noisy_files = [noisy_files[i] for i in idx]

    if args.model_type == "resflowse":
        print(f"\nLoading ResFlowSE: {args.ckpt}")
        model = ResFlowSEModel.load_from_checkpoint(
            args.ckpt, map_location="cpu", weights_only=False, strict=False
        )
        model.eval()
        if not args.no_ema:
            model._swap_to_ema()
        model.cuda()
        label = "ResFlowSE (1-NFE)"
        enhance_fn = lambda y, T: enhance_resflowse(model, y, T)

    else:  # flowse
        print(f"\nLoading FlowSE: {args.ckpt}")
        model = VFModel.load_from_checkpoint(
            args.ckpt, base_dir=args.data_dir, map_location="cpu"
        )
        for name, param in model.dnn.named_parameters():
            if name in model.ema_dnn:
                param.data = model.ema_dnn[name].to(param.device)
        model.eval()
        model.cuda()
        label = f"FlowSE ({args.N}-NFE)"
        enhance_fn = lambda y, T: enhance_flowse(model, y, T, N=args.N)

    results = evaluate_model(label, enhance_fn, clean_files, noisy_files, sess_sbo, sess_p808)
    print_results(label, results)

    if args.output:
        import json
        summary = {}
        for k, v in results.items():
            if v and k != "filename":
                arr = np.array(v)
                if np.issubdtype(arr.dtype, np.number):
                    summary[k] = {"mean": float(np.mean(arr)), "std": float(np.std(arr))}
        with open(args.output, "w") as f:
            json.dump({"label": label, "n_files": len(clean_files), "metrics": summary}, f, indent=2)
        print(f"\nResults saved to {args.output}")

    if args.output_csv:
        import pandas as pd
        df = pd.DataFrame({k: v for k, v in results.items() if v})
        df.to_csv(args.output_csv, index=False)
        print(f"Per-utterance results saved to {args.output_csv}")


if __name__ == "__main__":
    main()
