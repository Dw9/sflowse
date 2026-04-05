"""Waveform-domain losses for speech enhancement."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NegSISDRLoss(nn.Module):
    """Negative Scale-Invariant Signal-to-Distortion Ratio loss.

    Directly optimizes SI-SDR in the waveform domain. Standard loss in
    speech separation (Conv-TasNet, DPRNN, SepFormer).

    SI-SDR = 10 * log10(||s_target||² / ||e_noise||²)
    where s_target = (<x_hat, x> / ||x||²) * x, e_noise = x_hat - s_target.

    Returns negative SI-SDR (for minimization).
    """

    def forward(self, x_hat_wav, x_wav):
        """
        Args:
            x_hat_wav: [B, T] predicted waveform.
            x_wav: [B, T] target waveform.

        Returns:
            Scalar negative SI-SDR (mean over batch).
        """
        # Zero-mean normalization
        x_hat = x_hat_wav - x_hat_wav.mean(dim=-1, keepdim=True)
        x_ref = x_wav - x_wav.mean(dim=-1, keepdim=True)

        # s_target = projection of x_hat onto x_ref
        dot = (x_hat * x_ref).sum(dim=-1, keepdim=True)
        s_target = dot * x_ref / (x_ref.pow(2).sum(dim=-1, keepdim=True) + 1e-8)

        # e_noise = x_hat - s_target
        e_noise = x_hat - s_target

        si_sdr = 10 * torch.log10(
            s_target.pow(2).sum(dim=-1) / (e_noise.pow(2).sum(dim=-1) + 1e-8)
        )
        return -si_sdr.mean()


class MultiResolutionSTFTLoss(nn.Module):
    """Multi-resolution STFT loss (spectral convergence + log-magnitude L1 + optional complex L1).

    Computes STFT at multiple resolutions on waveforms, penalizing spectral
    differences at each scale. Standard in speech synthesis/enhancement
    (HiFi-GAN, Vocos, EnCodec).

    Args:
        resolutions: List of (n_fft, hop_length, win_length) tuples.
        factor_sc: Weight for spectral convergence loss.
        factor_mag: Weight for log-magnitude L1 loss.
        factor_complex: Weight for complex STFT L1 loss (real+imag, preserves phase).
    """

    DEFAULT_RESOLUTIONS = [
        (256, 64, 256),
        (512, 128, 512),
        (1024, 256, 1024),
        (2048, 512, 2048),
    ]

    def __init__(self, resolutions=None, factor_sc=1.0, factor_mag=1.0, factor_complex=0.0):
        super().__init__()
        self.resolutions = resolutions or self.DEFAULT_RESOLUTIONS
        self.factor_sc = factor_sc
        self.factor_mag = factor_mag
        self.factor_complex = factor_complex

        for n_fft, _, win_length in self.resolutions:
            self.register_buffer(f"window_{n_fft}", torch.hann_window(win_length))

    def _get_window(self, n_fft, device):
        window = getattr(self, f"window_{n_fft}")
        if window.device != device:
            window = window.to(device)
        return window

    def forward(self, x_hat_wav, x_wav):
        """Compute multi-resolution STFT loss.

        Args:
            x_hat_wav: [B, T] predicted waveform.
            x_wav: [B, T] target waveform.

        Returns:
            Scalar loss averaged over resolutions.
        """
        total_loss = 0.0
        for n_fft, hop_length, win_length in self.resolutions:
            window = self._get_window(n_fft, x_hat_wav.device)

            pred_stft = torch.stft(
                x_hat_wav, n_fft, hop_length=hop_length, win_length=win_length,
                window=window, return_complex=True,
            )
            target_stft = torch.stft(
                x_wav, n_fft, hop_length=hop_length, win_length=win_length,
                window=window, return_complex=True,
            )

            pred_mag = pred_stft.abs()
            target_mag = target_stft.abs()

            # Spectral convergence: Frobenius norm ratio
            sc_loss = torch.norm(target_mag - pred_mag, p="fro") / (
                torch.norm(target_mag, p="fro") + 1e-7
            )

            # Log-magnitude L1
            mag_loss = F.l1_loss(
                torch.log(pred_mag + 1e-7), torch.log(target_mag + 1e-7)
            )

            res_loss = self.factor_sc * sc_loss + self.factor_mag * mag_loss

            # Complex STFT L1 (phase-aware): penalizes real/imag differences
            if self.factor_complex > 0:
                complex_loss = F.l1_loss(pred_stft.real, target_stft.real) + \
                               F.l1_loss(pred_stft.imag, target_stft.imag)
                res_loss = res_loss + self.factor_complex * complex_loss

            total_loss = total_loss + res_loss

        return total_loss / len(self.resolutions)
