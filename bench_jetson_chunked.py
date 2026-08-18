"""bench_jetson_chunked.py — T-P2b: 分块推理设备侧 RTF(P2-1; SUBMIT_TP2b 提审版)
最优质量配置(2s 块 / 25% overlap / cross-fade, 源 p7_chunk_quality.json)在 Jetson 四档的 RTF。
分块逻辑逐行移植 p7_chunk_quality.enhance_chunked(不改现有文件; 入口 diff 已核:
load_audio=骨架版含 48k→16k resample, enhance_* 移植后与 p7 原文逐行比对)。

口径陷阱(任务书原文遵守):
- 分块计时 = 整个 enhance_chunked 的每文件墙钟 perf_counter(含循环/overlap-add/cross-fade/块间 STFT);
- 同会话同计时壳跑未分块基线(enhance_full 同计时), 只报同会话配对差与倍数;
- 禁止与 0.16246 跨口径直接比(那是 cuda.Event forward+iSTFT 口径)。

对拍(A100 先行, 独立脚本 chunked_parity_a100.py): 新移植 vs p7 原实现, 波形级 max|Δ|。
"""
import os, sys, json, hashlib, time, argparse, glob
import numpy as np, torch

sys.path.insert(0, os.path.expanduser("~/sflowse_pkg"))
os.environ['PATH'] = os.path.expanduser('~/.local/bin') + ':/usr/local/cuda/bin:' + os.environ['PATH']
from bench_jetson import env_snapshot, load_audio, prep_Y  # 骨架复用(load_audio 含 resample)

SR = 16000
CHUNK_S, OVERLAP = 2.0, 0.25   # 最优质量配置(p7_chunk_quality.json)


def enhance_full(model, y, T):
    """计时壳内: forward+iSTFT(cuda.Event 口径之'功能体', 但本任务计时=perf_counter 墙钟)。"""
    yn = y / y.abs().max()
    Y = pad_spec(torch.unsqueeze(model._forward_transform(model._stft(yn.cuda())), 0))
    with torch.no_grad():
        xh = model.forward(Y)
    return model.to_audio(xh.squeeze(), T).cpu().numpy()


def enhance_chunked(model, y, chunk_s, overlap):
    """逐行移植 p7_chunk_quality.enhance_chunked(唯一差异: torch.from_numpy 包装保持设备侧一致性)。"""
    C = int(chunk_s * SR); ov = int(overlap * C); hop = C - ov
    T = y.size(1); wav = y.squeeze()
    if T <= C:
        return enhance_full(model, y, T)
    out = torch.zeros(T); wsum = torch.zeros(T)
    win_edge = int(0.01 * SR)
    for start in range(0, T, hop):
        end = min(start + C, T)
        seg = wav[start:end]
        if seg.numel() < int(0.05 * SR):
            break
        seg3 = seg.unsqueeze(0)
        enh = torch.from_numpy(enhance_full(model, seg3, seg3.size(1)))
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


def pad_spec(Y):
    from flowmse.util.other import pad_spec as _ps
    return _ps(Y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode_id", type=int, required=True)
    ap.add_argument("--clocks", choices=["on", "off"], default="on")
    ap.add_argument("--n", type=int, default=824)
    ap.add_argument("--output", required=True)
    ap.add_argument("--energy", action="store_true", help="可选: tegrastats 能耗环(独立口径)")
    args = ap.parse_args()

    ENV_BEFORE = env_snapshot("before")
    os.system(f"echo {os.environ.get('JETSON_SUDO_PW','nx')} | sudo -S jetson_clocks --fan")
    nvp_q = os.popen("nvpmodel -q 2>/dev/null | tail -1").read().strip()
    print(f"power: {nvp_q} | cores: {os.cpu_count()}", flush=True)

    from flowmse.resflowse_model import ResFlowSEModel
    rfs = ResFlowSEModel.load_from_checkpoint(os.path.expanduser("~/sflowse.ckpt"), map_location="cpu",
                                              weights_only=False, strict=False)
    rfs.cuda().eval()

    noisy = sorted(glob.glob(os.path.expanduser("~/sflowse_pkg/data/test/noisy/*.wav")))[:args.n]
    loaded = [load_audio(nf) for nf in noisy]
    print(f"n={len(loaded)}", flush=True)

    # warmup(两路径都热: 块级小张量 + 整句大张量)
    for y, _ in loaded[:2]:
        enhance_full(rfs, y, y.size(1)); enhance_chunked(rfs, y, CHUNK_S, OVERLAP)

    files, dur, t_full, t_chunk = [], [], [], []
    for i, (y, d) in enumerate(loaded):
        T = y.size(1)
        t0 = time.perf_counter(); _ = enhance_chunked(rfs, y, CHUNK_S, OVERLAP); t1 = time.perf_counter()
        t2 = time.perf_counter(); _ = enhance_full(rfs, y, T); t3 = time.perf_counter()
        t_chunk.append((t1 - t0) * 1000); t_full.append((t3 - t2) * 1000); dur.append(d)
        files.append(os.path.basename(noisy[i]))
        if (i + 1) % 200 == 0:
            r = np.array(t_chunk) / 1000 / np.array(dur[:len(t_chunk)])
            print(f"[{i+1}/{len(loaded)}] chunked RTF running {r.mean():.4f}", flush=True)

    dur = np.array(dur); tf = np.array(t_full); tc = np.array(t_chunk)
    rtf_f = tf / 1000 / dur; rtf_c = tc / 1000 / dur
    ratio = tc / tf
    out = {
        "task": "T-P2b chunked inference RTF (2s/25%/crossfade)",
        "caliber": "perf_counter whole-function wall clock per file (chunk loop + overlap-add + crossfade included); paired same-session full-utterance baseline with the SAME timing shell; NOT comparable to cuda.Event caliber 0.16246",
        "config": {"chunk_s": CHUNK_S, "overlap": OVERLAP, "crossfade": "10ms edge ramp (p7 port)"},
        "power_mode": {"id": args.mode_id, "nvpmodel_q": nvp_q, "jetson_clocks": "on"},
        "n": len(loaded),
        "full_baseline": {"lat_ms_mean": float(tf.mean()), "rtf_mean": float(rtf_f.mean()), "rtf_p95": float(np.percentile(rtf_f, 95))},
        "chunked": {"lat_ms_mean": float(tc.mean()), "rtf_mean": float(rtf_c.mean()), "rtf_p95": float(np.percentile(rtf_c, 95)),
                     "rtf_p99": float(np.percentile(rtf_c, 99)), "p_ge_1": float(np.mean(rtf_c >= 1.0))},
        "paired_ratio": {"mean": float(ratio.mean()), "p95": float(np.percentile(ratio, 95)), "per_file_min": float(ratio.min()), "per_file_max": float(ratio.max())},
        "note": "offline full-utterance processing of a non-causal model; chunking here is a compute-batching strategy, NOT streaming",
        "env_before": ENV_BEFORE, "env_after": env_snapshot("after"),
        "perfile": {"file": files, "dur_s": list(map(float, dur)), "t_full_ms": list(map(float, tf)), "t_chunk_ms": list(map(float, tc))},
    }
    json.dump(out, open(os.path.expanduser(args.output), "w"), indent=2)
    np.save(os.path.expanduser(args.output).replace(".json", "_perfile.npy"),
            {"file": files, "dur_s": list(map(float, dur)), "t_full_ms": list(map(float, tf)), "t_chunk_ms": list(map(float, tc)),
             "rtf_full": list(map(float, rtf_f)), "rtf_chunk": list(map(float, rtf_c))})
    h = hashlib.sha256(open(os.path.expanduser(args.output), "rb").read()).hexdigest()
    open(os.path.expanduser(args.output) + ".sha256", "w").write(f"{h}  {os.path.basename(args.output)}\n")
    print(f"✓ saved sha={h[:16]} | full RTF {rtf_f.mean():.4f} chunked RTF {rtf_c.mean():.4f} ratio {ratio.mean():.3f}x", flush=True)


if __name__ == "__main__":
    main()
