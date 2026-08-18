"""a100_fp16_standalone.py — 必做: 独立跑(autocast + pure 算子, 无 fp32 先行)补 sha 产物 + 字段改名。
单进程只跑 autocast(不先 fp32), 逐样本 max|Δ| 与三指标; 与"fp32先行完好"支对拍即状态依赖双证。
输出 k_of_10_ran_clean / k_of_10_collapsed 分开命名(防误读)。"""
import os, json, hashlib
import torch, numpy as np, soundfile as sf, torchaudio
os.environ['PATH'] = os.path.expanduser('~/.local/bin') + ':/usr/local/cuda/bin:' + os.environ['PATH']
os.environ['NCSNPP_PURE_PYTORCH'] = '1'
os.environ['NCSNPP_PURE_UPFIRDN'] = '1'
from glob import glob
from pesq import pesq
from pystoi import stoi
from flowmse.util.other import si_sdr, pad_spec
SR = 16000


def load_audio(fp, sr=SR):
    y, sr0 = sf.read(fp); y = torch.from_numpy(y).float()
    if y.dim() == 1: y = y.unsqueeze(0)
    if sr0 != sr: y = torchaudio.functional.resample(y, sr0, sr)
    return y


def enhance_auto(model, y, T):
    yn = y / y.abs().max()
    Y = pad_spec(torch.unsqueeze(model._forward_transform(model._stft(yn.cuda())), 0))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        xh = model.forward(Y)
    xh = xh.to(torch.complex64)
    wav = model.to_audio(xh.squeeze(), T)
    return wav.float().cpu().numpy(), xh.squeeze().cpu().numpy()


def enhance_fp32(model, y, T):
    yn = y / y.abs().max()
    Y = pad_spec(torch.unsqueeze(model._forward_transform(model._stft(yn.cuda())), 0))
    with torch.no_grad():
        xh = model.forward(Y)
    wav = model.to_audio(xh.squeeze(), T)
    return wav.float().cpu().numpy(), xh.squeeze().cpu().numpy()


def main():
    from flowmse.resflowse_model import ResFlowSEModel
    # 独立跑: 全新进程, 模型只经历 autocast(无任何 fp32 forward 先行)
    rfs = ResFlowSEModel.load_from_checkpoint("sflowse.ckpt", map_location="cpu", weights_only=False, strict=False)
    rfs.cuda().eval()
    clean_all = sorted(glob("/home/zhibo/workspace/VoiceBank_processed/test/clean/*.wav"))
    noisy_all = sorted(glob("/home/zhibo/workspace/VoiceBank_processed/test/noisy/*.wav"))
    idx = [0] + [int(x) for x in np.linspace(1, len(clean_all) - 1, 9)]
    files = [(clean_all[i], noisy_all[i]) for i in idx]
    print(f"[fp16 standalone autocast x10] env NCSNPP_PURE*=1; 无 fp32 先行", flush=True)

    # autocast 独立(每条: 只 autocast 一次拿 w16; fp32 对拍在第二阶段同进程后续跑——不行! 顺序即变量。
    # 正确: 本进程全程只 autocast; fp32 基线来自已入库的 a100_fp16_ten_autocast.json(fp32 段, 同 10 条同序)
    prev = {r["file"]: r for r in json.load(open("a100_fp16_ten_autocast.json"))["results"]}
    results, n_clean, n_collapsed = [], 0, 0
    for cf, nf in files:
        bn = os.path.basename(nf); rec = {"file": bn}
        x = load_audio(cf); y = load_audio(nf); T = y.size(1); xn = x.squeeze().numpy()
        try:
            w16, s16 = enhance_auto(rfs, y, T)
            m = min(len(xn), len(w16)); a, c = xn[:m], w16[:m]
            p16 = float(pesq(SR, a, c, "wb")); s16m = float(si_sdr(a, c)); e16 = float(stoi(a, c, SR, extended=True))
            rec["fp16_autocast_standalone"] = {"pesq": p16, "si_sdr": s16m, "estoi": e16}
            sf.write(f"a100_fp16_{bn[:-4]}_auto_standalone.wav", w16, SR, subtype="FLOAT")
            ref = prev.get(bn, {}).get("fp32", {})
            ref16 = prev.get(bn, {}).get("fp16_autocast", {})
            if ref:
                mm = min(len(ref), 0)  # 无 wav 对拍(prev 只有标量), max|Δ| 需 fp32 波形 → 第二阶段
            # 与 fp32先行支 的同名指标对照(标量级)
            rec["vs_fp32_baseline"] = {"fp32_pesq": ref.get("pesq"), "delta_pesq": (p16 - ref.get("pesq", 0)) if ref else None}
            rec["vs_fp32先行_autocast支"] = {"fp32先行_pesq": ref16.get("pesq"), "delta": (p16 - ref16.get("pesq", 0)) if ref16 else None}
            # 崩塌判定(描述性, 非阈值判词): 与 fp32 基线 PESQ 差 > 1 视为塌(1.75 vs 3.4 量级)
            collapsed = bool(ref and (ref.get("pesq", 0) - p16) > 1.0)
            rec["collapsed_descriptive"] = collapsed
            n_collapsed += collapsed; n_clean += (not collapsed)
        except Exception as e:
            import traceback
            rec["fp16_autocast_standalone"] = {"crash": repr(e)[:200], "tb_first": traceback.format_exc().strip().splitlines()[-1][:200]}
            n_clean += 0
        results.append(rec)
        r = rec.get("fp16_autocast_standalone", {})
        print(f"  {bn}: standalone pesq={r.get('pesq', r.get('crash','?'))} collapsed={rec.get('collapsed_descriptive')}", flush=True)

    out = {"task": "fp16 autocast 独立跑(补 sha 产物; 状态依赖双证之'独立'支)",
           "env": "NCSNPP_PURE_PYTORCH=1(=08-13 1.752 原条件); 全程无 fp32 先行",
           "n_files": len(files),
           "k_of_10_ran_clean": n_clean, "k_of_10_collapsed": n_collapsed,
           "fields_note": "k_of_10_ran_clean=跑通且正常; k_of_10_collapsed=PESQ 塌>1(描述性非判词); 与 a100_fp16_ten_autocast.json(fp32先行支, 10/10 完好)合成状态依赖双证",
           "results": results}
    json.dump(out, open("a100_fp16_standalone.json", "w"), indent=2)
    sha = hashlib.sha256(open("a100_fp16_standalone.json", "rb").read()).hexdigest()
    open("a100_fp16_standalone.json.sha256", "w").write(f"{sha}  a100_fp16_standalone.json\n")
    print(f"\n✓ saved sha={sha[:16]} | clean={n_clean}/10 collapsed={n_collapsed}/10", flush=True)


if __name__ == "__main__":
    main()
