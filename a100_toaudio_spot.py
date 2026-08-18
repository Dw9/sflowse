"""a100_toaudio_bench.py — A100 to_audio 全824 直计(REVIEW 5条件版; SUBMIT_a100_dual PASS)
闭合 PLAN §3.4 假设(2), 判读按 1.2: **差分残差(0.642)是 iSTFT 成本的上界**(残差=iSTFT+一切不随N增长的
per-call 开销), 直计 <= 0.642 是预期; 二者之差 = 其余 per-call 开销。直计给 iSTFT 占多少。
口径: cuda.Event; Y 预计算; batch=1; 每句重复 ≥5 次取中位数(1.3); warmup 10 句不计。
对照(1.1 出处写明): 0.642 出自 **FlowSE** N1/N2 差分(per_step=39.818, 反推 N1=40.460),
非主表 ResFlowSE 40.803; to_audio 同函数可比但须注明。"""
import os, json, time, hashlib
import torch, numpy as np, soundfile as sf, torchaudio
os.environ['PATH'] = os.path.expanduser('~/.local/bin') + ':/usr/local/cuda/bin:' + os.environ['PATH']
from glob import glob
SR = 16000
NREP = 7  # 每句重复次数取中位(≥5)


def load_audio(fp, sr=SR):
    y, sr0 = sf.read(fp); y = torch.from_numpy(y).float()
    if y.dim() == 1: y = y.unsqueeze(0)
    if sr0 != sr: y = torchaudio.functional.resample(y, sr0, sr)
    return y


def stats(a):
    a = np.asarray(a, float)
    return {"n": int(len(a)), "mean": float(a.mean()), "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)), "p99": float(np.percentile(a, 99)), "std": float(a.std())}


def main():
    from flowmse.util.other import pad_spec
    from flowmse.resflowse_model import ResFlowSEModel
    device = "cuda"
    rfs = ResFlowSEModel.load_from_checkpoint("sflowse.ckpt", map_location="cpu", weights_only=False, strict=False)
    rfs.cuda().eval()
    noisy = sorted(glob("/home/zhibo/workspace/VoiceBank_processed/test/noisy/*.wav"))
    durs = np.array([sf.info(f).frames / sf.info(f).samplerate for f in noisy])
    print(f"[A100 to_audio] n={len(noisy)} nrep={NREP}(median)", flush=True)

    def prep(x):
        yn = x / x.abs().max()
        return pad_spec(torch.unsqueeze(rfs._forward_transform(rfs._stft(yn.to(device))), 0))

    med_lat = []
    for i, f in enumerate(noisy):
        y = load_audio(f); T = y.size(1); Y = prep(y)
        with torch.no_grad():
            xh = rfs.forward(Y)
        spec = xh.squeeze()
        ts = []
        for _ in range(NREP):
            s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
            s.record(); rfs.to_audio(spec, T); e.record(); torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
        med_lat.append(np.median(ts))
        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/824] median lat mean-so-far {np.mean(med_lat):.4f}ms", flush=True)
        if i == 9:  # warmup 前10句不计
            med_lat = []
    med_lat = np.array(med_lat)

    direct = float(med_lat.mean())
    diff_flowse = 0.642  # bench_a100 decomposition: iSTFT_ms_approx = N1-(N2-N1), FlowSE 差分
    out = {
        "task": "A100 to_audio 全824 直计(§3.4 假设2 闭合)",
        "caliber": "cuda.Event, 每句 %d 次取中位, Y 预计算, batch=1, warmup 10 句不计" % NREP,
        "n": int(len(med_lat)),
        "to_audio_latency_ms": stats(med_lat),
        "comparison": {
            "A100_direct_mean_ms": direct,
            "A100_differential_residual_ms": diff_flowse,
            "differential_provenance_1_1": "0.642 出自 FlowSE N1/N2 差分(per_step=39.818, 反推 N1=40.460) — bench_a100.json decomposition.iSTFT_ms_approx; 非主表 ResFlowSE 40.803(勿用 40.803-39.818=0.985 对照)",
            "interpretation_1_2": "差分残差是 iSTFT 成本的上界(= iSTFT + 一切不随N增长的 per-call 开销); 直计 <= 0.642 为预期非证伪; 二者之差 = 其余 per-call 开销",
            "iSTFT_share_of_residual_pct": float(direct / diff_flowse * 100),
            "other_percall_overhead_ms": float(diff_flowse - direct),
            "Orin_direct_ref_ms": 0.5522,
            "Orin_vs_A100_direct_note": "跨硬件比值仅参考(不同机不同CPU); Orin 1.06(25W)/0.55(MAXN) 见各档 json",
        },
    }
    json.dump(out, open("a100_toaudio_spot.json", "w"), indent=2)
    sha = hashlib.sha256(open("a100_toaudio_spot.json", "rb").read()).hexdigest()
    open("a100_toaudio_spot.json.sha256", "w").write(f"{sha}  a100_toaudio_spot.json\n")
    print(json.dumps(out["comparison"], indent=2, ensure_ascii=False))
    print(f"✓ saved  sha={sha[:16]}", flush=True)


if __name__ == "__main__":
    main()
