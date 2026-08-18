"""chunked_parity_a100.py — T-P2b 对拍先行(铁律2): 新移植 bench_jetson_chunked.enhance_chunked
vs p7_chunk_quality.enhance_chunked 原实现, 波形级 max|Δ|(10 文件, A100)。
通过判据: max|Δ| == 0(逐位)或 < 1e-6(浮点顺序差);否则 FAIL 不上板。
"""
import sys
import numpy as np
import torch

sys.path.insert(0, ".")
import importlib.util

def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

p7 = load_mod("p7", "p7_chunk_quality.py")
bc = load_mod("bc", "bench_jetson_chunked.py")

SR = 16000
from flowmse.resflowse_model import ResFlowSEModel
rfs = ResFlowSEModel.load_from_checkpoint("sflowse.ckpt", map_location="cpu", weights_only=False, strict=False)
rfs.cuda().eval()

import glob, soundfile as sf
files = sorted(glob.glob("/home/zhibo/workspace/VoiceBank_processed/test/noisy/*.wav"))
idx = [0] + [int(v) for v in np.linspace(1, len(files) - 1, 9)]
worst = 0.0
for i in idx:
    y, sr0 = sf.read(files[i], dtype="float32")
    y = torch.from_numpy(y).float().unsqueeze(0)
    if sr0 != SR:
        import torchaudio
        y = torchaudio.functional.resample(y, sr0, SR)
    a = p7.enhance_chunked(rfs, y, 2.0, 0.25)
    b = bc.enhance_chunked(rfs, y, 2.0, 0.25)
    d = float(np.abs(a - b).max()) if a.shape == b.shape else float("inf")
    worst = max(worst, d)
    print(f"{i}: max|Δ| = {d:.2e} {'OK' if d <= 1e-6 else 'FAIL'}", flush=True)
print(f"\nWORST = {worst:.2e} → {'PASS(上板)' if worst <= 1e-6 else 'FAIL(不上板)'}")
sys.exit(0 if worst <= 1e-6 else 1)
