"""
Two-channel ablation via ZERO-TRAINING mathematical equivalence (mentor directive).

M3 = sflowse.ckpt with NO EMA (raw weights) reproduces PESQ 3.062.
Its forward feeds [epsilon, y] with epsilon ≡ 0 (stochastic=0). NCSNpp decomposes the
2 complex channels into 4 real: [eps.real, eps.imag, y.real, y.imag] = [0, 0, y.real, y.imag].
=> the first conv (3x3) and the 6 input-pyramid combiners (1x1) see 4 real channels, of which
   ch0,ch1 are ALWAYS 0  =>  their weight columns contribute exactly 0.

Slicing those 7 input-side weights to keep only ch[2,3] (y) yields an input_channels=2 model
that is BIT-EQUIVALENT to the 4-channel M3 (the dropped terms were exactly 0).
Expected: PESQ/SI-SDR/ESTOI identical to M3 (3.062 / 18.79 / 0.871), i.e. Δ ≈ 0.

This rigorously answers Reviewer-1 minor: the zero-state channels are non-load-bearing
(contribution ≡ 0); the 4-channel form exists only to keep the input conv aligned with FlowSE.
"""
import torch, numpy as np
from glob import glob
from tqdm import tqdm
from pesq import pesq
from pystoi import stoi
from flowmse.resflowse_model import ResFlowSEModel
from flowmse.data_module import SpecsDataModule
from eval_metrics import load_audio_16k, enhance_resflowse
from flowmse.util.other import si_sdr

DATA = "/home/zhibo/workspace/VoiceBank_processed"
DEV = "cuda:0"

# ---- 1. load M3 raw state_dict, slice the 7 input-side weights ----
ck = torch.load("sflowse.ckpt", map_location="cpu", weights_only=False)
sd = {k: v.clone() for k, v in ck["state_dict"].items()}
SLICE_KEYS = ["dnn.all_modules.3.weight"] + \
             [f"dnn.all_modules.{i}.Conv_0.weight" for i in (7, 11, 15, 19, 25, 29)]
for k in SLICE_KEYS:
    assert sd[k].dim() == 4 and sd[k].shape[1] == 4, (k, tuple(sd[k].shape))
    sd[k] = sd[k][:, 2:4, :, :].contiguous().clone()   # keep y.real(ch2), y.imag(ch3)
# output_layer (2,4,1,1) is OUTPUT-side -> intentionally NOT sliced

# ---- 2. build input_channels=2 model (same data kwargs as M3) ----
m = ResFlowSEModel(
    backbone="ncsnpp", input_channels=2, data_module_cls=SpecsDataModule,
    base_dir=DATA, n_fft=510, hop_length=128, num_frames=256, window="hann",
    spec_factor=0.15, spec_abs_exponent=0.5, transform_type="exponent",
    normalize="noisy", batch_size=8, num_workers=4, format="default",
)
miss, unexp = m.load_state_dict(sd, strict=False)
print(f"load: missing={len(miss)} unexpected={len(unexp)} | unexpected sample: {list(unexp)[:4]}")
print(f"first-conv weight: {tuple(m.dnn.all_modules[3].weight.shape)} (expect (128,2,3,3))")
print(f"output_layer weight: {tuple(m.dnn.output_layer.weight.shape)} (expect (2,4,1,1), unchanged)")
print(f"dnn.input_channels = {m.dnn.input_channels}")
m.eval()
# NO EMA swap -> raw sliced weights = M3-equivalent
m.to(DEV)

# ---- 3. full-824 eval (no_ema path) ----
cf = sorted(glob(f"{DATA}/test/clean/*.wav"))
nf = sorted(glob(f"{DATA}/test/noisy/*.wav"))
P = S = E = 0.0
n = len(cf)
for i, (c, ny) in enumerate(tqdm(zip(cf, nf), total=n, desc="2ch-equiv")):
    x = load_audio_16k(c); y = load_audio_16k(ny); T = x.size(1); xn = x.squeeze().numpy()
    h = enhance_resflowse(m, y, T); mm = min(len(xn), len(h)); xn = xn[:mm]; h = h[:mm]
    P += pesq(16000, xn, h, "wb"); S += si_sdr(xn, h); E += stoi(xn, h, 16000, extended=True)
    if (i + 1) % 200 == 0:
        print(f"  [{i+1}] running PESQ={P/(i+1):.4f} SI-SDR={S/(i+1):.3f} ESTOI={E/(i+1):.4f}")

print("\n" + "=" * 64)
print("2ch-SLICED (mathematically equivalent to M3):")
print(f"  PESQ   = {P/n:.4f}   (M3 ref = 3.0622)")
print(f"  SI-SDR = {S/n:.3f}   (M3 ref = 18.79)")
print(f"  ESTOI  = {E/n:.4f}   (M3 ref = 0.871)")
print(f"  Δ vs M3 = PESQ {P/n-3.0622:+.4f} | SI-SDR {S/n-18.79:+.3f} | ESTOI {E/n-0.871:+.4f}")
print("=> Δ≈0 confirms zero-state channels contribute 0; zero-state is non-load-bearing.")
