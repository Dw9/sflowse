"""
Speed benchmark: ResFlowSE (1-NFE) vs FlowSE (5-NFE).
Measures inference time, RTF (Real-Time Factor), and throughput on full test set.

Usage:
  python benchmark_speed.py \
    --resflowse_ckpt <path> \
    --flowse_ckpt <path> \
    --data_dir /home/zhibo/workspace/VoiceBank_processed
"""

import argparse
import time
import torch
import torch.serialization
import numpy as np
import soundfile as sf
import torchaudio
from glob import glob
import os

from flowmse.util.other import pad_spec

SR = 16000


def load_audio(filepath, target_sr=16000):
    waveform, sample_rate = sf.read(filepath)
    waveform = torch.from_numpy(waveform).float()
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if sample_rate != target_sr:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sr)
    return waveform


def benchmark_resflowse(ckpt, noisy_files, N=1, device="cuda"):
    from flowmse.resflowse_model import ResFlowSEModel

    model = ResFlowSEModel.load_from_checkpoint(
        ckpt, map_location="cpu", weights_only=False, strict=False
    )
    model.eval()
    model.to(device)

    # Warmup (3 files)
    for nf in noisy_files[:3]:
        y = load_audio(nf, SR)
        y_norm = y / y.abs().max()
        Y = torch.unsqueeze(model._forward_transform(model._stft(y_norm.to(device))), 0)
        Y = pad_spec(Y)
        with torch.no_grad():
            x_hat = Y
            for _ in range(N):
                x_hat = model.forward(x_hat)
            _ = model.to_audio(x_hat.squeeze(), y.size(1))
    torch.cuda.synchronize()

    times = []
    audio_durations = []
    for i, nf in enumerate(noisy_files):
        y = load_audio(nf, SR)
        T_orig = y.size(1)
        audio_durations.append(T_orig / SR)

        norm_factor = y.abs().max()
        y_norm = y / norm_factor
        Y = torch.unsqueeze(model._forward_transform(model._stft(y_norm.to(device))), 0)
        Y = pad_spec(Y)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            x_hat = Y
            for _ in range(N):
                x_hat = model.forward(x_hat)
            _ = model.to_audio(x_hat.squeeze(), T_orig)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        times.append(t1 - t0)
        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{len(noisy_files)}]")

    return np.array(times), np.array(audio_durations)


def benchmark_flowse(ckpt, data_dir, device="cuda", N=5):
    from flowmse.data_module import SpecsDataModule, load_audio as flowse_load_audio
    from flowmse.model import VFModel
    from flowmse.sampling import get_white_box_solver

    torch.serialization.add_safe_globals([SpecsDataModule])

    model = VFModel.load_from_checkpoint(
        ckpt, base_dir=data_dir, map_location="cpu"
    )
    model.data_module.setup(stage=None)

    # Swap to EMA weights
    for name, param in model.dnn.named_parameters():
        if name in model.ema_dnn:
            param.data = model.ema_dnn[name].to(param.device)

    model.eval()
    model.to(device)

    T_rev = model.T_rev
    t_eps = model.t_eps
    noisy_files = model.data_module.test_set.noisy_files

    # Warmup (3 files)
    for nf in noisy_files[:3]:
        y, _ = flowse_load_audio(nf)
        y_norm = y / y.abs().max()
        Y = torch.unsqueeze(model._forward_transform(model._stft(y_norm.to(device))), 0)
        Y = pad_spec(Y)
        with torch.no_grad():
            sampler = get_white_box_solver(
                "euler", model.ode, model, Y.to(device),
                T_rev=T_rev, t_eps=t_eps, N=N
            )
            sample, _ = sampler()
            _ = model.to_audio(sample.squeeze(), y.size(1))
    torch.cuda.synchronize()

    times = []
    audio_durations = []
    for i, nf in enumerate(noisy_files):
        y, _ = flowse_load_audio(nf)
        T_orig = y.size(1)
        audio_durations.append(T_orig / SR)

        norm_factor = y.abs().max()
        y_norm = y / norm_factor
        Y = torch.unsqueeze(model._forward_transform(model._stft(y_norm.to(device))), 0)
        Y = pad_spec(Y)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            sampler = get_white_box_solver(
                "euler", model.ode, model, Y.to(device),
                T_rev=T_rev, t_eps=t_eps, N=N
            )
            sample, _ = sampler()
            _ = model.to_audio(sample.squeeze(), T_orig)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        times.append(t1 - t0)
        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{len(noisy_files)}]")

    return np.array(times), np.array(audio_durations), noisy_files


def print_report(d_times, d_audio, f_times, f_audio, n_files):
    """Print and return formatted benchmark report."""
    d_rtfs = d_times / d_audio
    f_rtfs = f_times / f_audio

    lines = []
    def p(s=""):
        lines.append(s)
        print(s)

    p("=" * 70)
    p("  INFERENCE SPEED BENCHMARK — ResFlowSE vs FlowSE")
    p("  VoiceBank test set, RTX 4090, PyTorch 2.x, 16kHz")
    p("=" * 70)
    p()

    # ResFlowSE
    p("┌─────────────────────────────────────────────────────────────────┐")
    p("│  ResFlowSE (1-NFE, single-step)                              │")
    p("├─────────────────────────────────────────────────────────────────┤")
    p(f"│  Files evaluated:     {n_files:>6d}                                  │")
    p(f"│  Total audio:         {d_audio.sum():>8.1f}s  ({d_audio.sum()/60:.1f} min)                  │")
    p(f"│  Total inference:     {d_times.sum():>8.3f}s                              │")
    p(f"│  Avg per file:        {d_times.mean()*1000:>8.2f} ms (±{d_times.std()*1000:.2f} ms)           │")
    p(f"│  Median per file:     {np.median(d_times)*1000:>8.2f} ms                          │")
    p(f"│  RTF (mean):          {d_rtfs.mean():>8.5f}                              │")
    p(f"│  RTF (median):        {np.median(d_rtfs):>8.5f}                              │")
    p(f"│  RTF (p95):           {np.percentile(d_rtfs, 95):>8.5f}                              │")
    p(f"│  Real-time factor:    {1/d_rtfs.mean():>8.1f}x faster than real-time       │")
    p("└─────────────────────────────────────────────────────────────────┘")
    p()

    # FlowSE
    p("┌─────────────────────────────────────────────────────────────────┐")
    p("│  FlowSE (5-NFE, Euler ODE solver)                             │")
    p("├─────────────────────────────────────────────────────────────────┤")
    p(f"│  Files evaluated:     {n_files:>6d}                                  │")
    p(f"│  Total audio:         {f_audio.sum():>8.1f}s  ({f_audio.sum()/60:.1f} min)                  │")
    p(f"│  Total inference:     {f_times.sum():>8.3f}s                              │")
    p(f"│  Avg per file:        {f_times.mean()*1000:>8.2f} ms (±{f_times.std()*1000:.2f} ms)           │")
    p(f"│  Median per file:     {np.median(f_times)*1000:>8.2f} ms                          │")
    p(f"│  RTF (mean):          {f_rtfs.mean():>8.5f}                              │")
    p(f"│  RTF (median):        {np.median(f_rtfs):>8.5f}                              │")
    p(f"│  RTF (p95):           {np.percentile(f_rtfs, 95):>8.5f}                              │")
    p(f"│  Real-time factor:    {1/f_rtfs.mean():>8.1f}x faster than real-time       │")
    p("└─────────────────────────────────────────────────────────────────┘")
    p()

    # Comparison
    speedup = f_times.sum() / d_times.sum()
    rtf_ratio = f_rtfs.mean() / d_rtfs.mean()
    p("┌─────────────────────────────────────────────────────────────────┐")
    p("│  COMPARISON                                                    │")
    p("├─────────────────────────────────────────────────────────────────┤")
    p(f"│  Inference speedup:   {speedup:>8.2f}x  (ResFlowSE vs FlowSE)       │")
    p(f"│  RTF ratio:           {rtf_ratio:>8.2f}x                              │")
    p(f"│  ResFlowSE RTF:      {d_rtfs.mean():.5f}  →  {1/d_rtfs.mean():.0f}x real-time            │")
    p(f"│  FlowSE RTF:          {f_rtfs.mean():.5f}  →  {1/f_rtfs.mean():.0f}x real-time             │")
    p("└─────────────────────────────────────────────────────────────────┘")
    p()

    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resflowse_ckpt", type=str, required=True)
    parser.add_argument("--flowse_ckpt", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="/home/zhibo/workspace/VoiceBank_processed")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output", type=str, default=None,
                        help="Save report to file")
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"

    # Get noisy files for ResFlowSE
    noisy_dir = os.path.join(args.data_dir, args.split, "noisy")
    noisy_files = sorted(glob(os.path.join(noisy_dir, "*.wav")))
    n_files = len(noisy_files)
    print(f"Benchmarking on {n_files} files from {args.split} split\n")

    # ResFlowSE
    print(">>> Running ResFlowSE (1-NFE) ...")
    d_times, d_audio = benchmark_resflowse(args.resflowse_ckpt, noisy_files, N=1, device=device)
    torch.cuda.empty_cache()

    # FlowSE
    print("\n>>> Running FlowSE (5-NFE) ...")
    f_times, f_audio, _ = benchmark_flowse(args.flowse_ckpt, args.data_dir, device=device, N=5)

    # Report
    report = print_report(d_times, d_audio, f_times, f_audio, n_files)

    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"Report saved to {args.output}")
