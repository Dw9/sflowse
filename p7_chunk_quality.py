"""p7_chunk_quality.py — P7 质量侧(A100): 分块推理的质量代价(PLAN_FINAL §P7)。
chunk C ∈ {0.5, 1, 2}s × overlap {0, 25%}, 全 824, ResFlowSE(sflowse, 确定性), 对照整句(非分块)。
分块法: 波形滑窗(Hann 窗淡化拼接, overlap 段 cross-fade) → 每块独立 STFT+forward+iSTFT → 拼回波形 → 三指标。
⚠️ 范围: 不声称流式(模型非因果, 分块是权宜); 只给 算法延迟=C+计算时间 与 ΔPESQ 有界曲线。
输出: p7_chunk_quality.json + .sha256(仓库根)。fp16 dtype 搭便车在另一脚本。"""
import os, json, hashlib
import torch, numpy as np, soundfile as sf, torchaudio
os.environ['PATH'] = os.path.expanduser('~/.local/bin') + ':/usr/local/cuda/bin:' + os.environ['PATH']
from glob import glob
from pesq import pesq
from pystoi import stoi
from flowmse.util.other import si_sdr, pad_spec
SR = 16000
CHUNKS = [0.5, 1.0, 2.0]      # s
OVERLAPS = [0.0, 0.25]        # 比例


def load_audio(fp, sr=SR):
    y, sr0 = sf.read(fp); y = torch.from_numpy(y).float()
    if y.dim() == 1: y = y.unsqueeze(0)
    if sr0 != sr: y = torchaudio.functional.resample(y, sr0, sr)
    return y


def enhance_full(model, y, T):
    yn = y / y.abs().max()
    Y = pad_spec(torch.unsqueeze(model._forward_transform(model._stft(yn.cuda())), 0))
    with torch.no_grad():
        xh = model.forward(Y)
    return model.to_audio(xh.squeeze(), T).cpu().numpy()


def enhance_chunked(model, y, chunk_s, overlap):
    """波形域滑窗分块: 块长 C, 重叠 overlap*C, 块内完整 STFT+forward+iSTFT, 拼接(cross-fade)。"""
    C = int(chunk_s * SR)
    ov = int(overlap * C)
    hop = C - ov
    T = y.size(1); wav = y.squeeze()
    if T <= C:
        return enhance_full(model, y, T)
    out = torch.zeros(T); wsum = torch.zeros(T)
    win_edge = int(0.01 * SR)  # 10ms 边缘 ramp(块端点非整周期); overlap>0 时用 cross-fade
    for start in range(0, T, hop):
        end = min(start + C, T)
        seg = wav[start:end]
        if seg.numel() < int(0.05 * SR):   # <50ms 尾块并入前块逻辑: 直接跳过(前块已覆盖)
            break
        seg3 = seg.unsqueeze(0)
        enh = torch.from_numpy(enhance_full(model, seg3, seg3.size(1)))
        # 权重窗: overlap>0 → Hann 端部 cross-fade; overlap=0 → 常数(块边界可能 click, 如实)
        if ov > 0:
            w = torch.ones(end - start)
            ramp = min(win_edge, (end - start) // 2)
            if start > 0:
                w[:ramp] = torch.linspace(0, 1, ramp)
            if end < T:
                w[-ramp:] = torch.linspace(1, 0, ramp)
        else:
            w = torch.ones(end - start)
        out[start:end] += enh * w; wsum[start:end] += w
    out = out / torch.clamp(wsum, min=1e-8)
    return out.numpy()


def main():
    from flowmse.resflowse_model import ResFlowSEModel
    rfs = ResFlowSEModel.load_from_checkpoint("sflowse.ckpt", map_location="cpu", weights_only=False, strict=False)
    rfs.cuda().eval()
    clean = sorted(glob("/home/zhibo/workspace/VoiceBank_processed/test/clean/*.wav"))
    noisy = sorted(glob("/home/zhibo/workspace/VoiceBank_processed/test/noisy/*.wav"))
    print(f"[P7] n={len(noisy)}; chunks={CHUNKS}s overlaps={OVERLAPS}", flush=True)

    results = {}
    for tag, fn in [("full", None)] + [(f"C{c:g}s_ov{int(o*100)}", (c, o)) for c in CHUNKS for o in OVERLAPS]:
        pesqs, sdrs, estois = [], [], []
        for i, (cf, nf) in enumerate(zip(clean, noisy)):
            x = load_audio(cf).squeeze().numpy(); y = load_audio(nf); T = y.size(1)
            xh = enhance_full(rfs, y, T) if fn is None else enhance_chunked(rfs, y, fn[0], fn[1])
            m = min(len(x), len(xh)); a, b = x[:m], xh[:m]
            pesqs.append(pesq(SR, a, b, "wb")); sdrs.append(float(si_sdr(a, b))); estois.append(stoi(a, b, SR, extended=True))
            if (i + 1) % 200 == 0:
                print(f"  [{tag} {i+1}/824] PESQ~{np.mean(pesqs):.4f}", flush=True)
        results[tag] = {"pesq": float(np.mean(pesqs)), "si_sdr": float(np.mean(sdrs)), "estoi": float(np.mean(estois))}
        print(f"  ✓ {tag}: PESQ {results[tag]['pesq']:.4f} SI-SDR {results[tag]['si_sdr']:.3f} ESTOI {results[tag]['estoi']:.4f}", flush=True)

    full = results["full"]
    curves = {t: {"delta_pesq": v["pesq"] - full["pesq"], "delta_si_sdr": v["si_sdr"] - full["si_sdr"],
                  "delta_estoi": v["estoi"] - full["estoi"], **v} for t, v in results.items() if t != "full"}
    out = {"task": "P7 分块推理质量代价(A100; PLAN §P7)",
           "scope_note": "不声称流式(模型非因果, 分块权宜); 算法延迟=C+计算时间; 正解是因果架构(Stream.FM)本文不做——正文明说",
           "n": len(noisy), "model": "ResFlowSE(sflowse, 确定性)", "chunking": "波形滑窗; ov>0 用 10ms 端 ramp cross-fade; ov=0 块边界直接拼",
           "full_utterance_ref": full, "chunk_results": results, "delta_vs_full": curves,
           "algo_latency_note": "每配置算法延迟下界 = C(块等待)+ 计算; 实测每块计算时间在 Jetson 复用 bench(质量与硬件无关, parity 6.89e-4 旧/3.34e-4 v5)"}
    json.dump(out, open("p7_chunk_quality.json", "w"), indent=2)
    sha = hashlib.sha256(open("p7_chunk_quality.json", "rb").read()).hexdigest()
    open("p7_chunk_quality.json.sha256", "w").write(f"{sha}  p7_chunk_quality.json\n")
    print("\nΔPESQ vs full: " + json.dumps({t: round(v["delta_pesq"], 4) for t, v in curves.items()}, indent=1), flush=True)
    print(f"✓ saved sha={sha[:16]}", flush=True)


if __name__ == "__main__":
    main()
