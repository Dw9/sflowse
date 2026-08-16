"""gtcrn_quality_vb.py — GTCRN 官方 PyTorch vctk.tar 在 VB-DMD 824 测试集的质量评测(导师裁定: 测, 两值并列)。
五硬约束: 只用 vctk.tar(sha a0f0e044..); 走 eval_dns 同管线同批 824 同指标(PESQ/SI-SDR/ESTOI/DNSMOS/P808);
实测与 published 并列报差; 若明显低于 published 先排查预处理错配(归一化/重采样/dtype/尺度)排查不掉如实记未解决差异
不作 GTCRN 结论; 锚点非竞赛禁输赢。PLAN §7 预注册: 负面答案(小参数量拿大部分质量)可接受, 出数不改口。
管线对齐: 官方 infer.py 流程(STFT 512/256/hann^0.5, 模型, iSTFT); 加载路径同 eval_dns(48k→resample16k,
sf.read 实际 sr); 无归一化(官方对 raw 波形操作)。"""
import os, sys, json, csv, hashlib
import torch, numpy as np, soundfile as sf, torchaudio
sys.path.insert(0, "third_party/gtcrn")
os.environ['PATH'] = os.path.expanduser('~/.local/bin') + ':/usr/local/cuda/bin:' + os.environ['PATH']
from glob import glob
from pesq import pesq
from pystoi import stoi
from eval_metrics import load_dnsmos, compute_dnsmos_single
from flowmse.util.other import si_sdr
SR = 16000
CKPT = "third_party/gtcrn/checkpoints/model_trained_on_vctk.tar"
PUBLISHED = {"pesq": 2.792, "si_sdr": None, "estoi": 0.80, "macs_per_s_G": 0.033, "params_M": 0.0482,
             "source": "GTCRN ICASSP2024 paper Table 1 (VCTK-DEMAND); si_sdr/estoi 以论文表为准此处占位待核"}


def load16k(fp):
    y, sr0 = sf.read(fp, dtype='float32')
    y = torch.from_numpy(y).float()
    if y.dim() == 1: y = y.unsqueeze(0)
    if sr0 != SR: y = torchaudio.functional.resample(y, sr0, SR)
    return y


def main():
    from gtcrn import GTCRN
    device = "cuda"
    m = GTCRN().eval()
    ck = torch.load(CKPT, map_location="cpu")
    m.load_state_dict(ck["model"]); m.cuda()
    sha = hashlib.sha256(open(CKPT, "rb").read()).hexdigest()
    print(f"[GTCRN vb] ckpt sha256 {sha[:16]}...", flush=True)

    clean = sorted(glob("/home/zhibo/workspace/VoiceBank_processed/test/clean/*.wav"))
    noisy = sorted(glob("/home/zhibo/workspace/VoiceBank_processed/test/noisy/*.wav"))
    assert len(clean) == len(noisy) == 824
    # 行序对齐 m3.csv(与 eval_dns 全表同序)
    m3fn = [r["filename"] for r in csv.DictReader(open("eval_dns/m3.csv"))]
    cbn = [os.path.basename(c) for c in clean]
    assert cbn == m3fn, "glob 顺序 != m3.csv"

    sbo, p808s = load_dnsmos("/home/zhibo/.torchmetrics/DNSMOS/DNSMOS")
    rows = []
    for i, (cf, nf) in enumerate(zip(clean, noisy)):
        x = load16k(cf).squeeze().numpy(); y = load16k(nf).squeeze()
        T = y.numel()
        spec = torch.stft(y, 512, 256, 512, torch.hann_window(512).pow(0.5), return_complex=False).cuda()
        with torch.no_grad():
            out = m(spec[None])[0].cpu()
        # torch 2.x istft 要 complex(官方代码是 1.11 的 return_complex=False 风格): view_as_real → complex
        enh = torch.istft(torch.view_as_complex(out.contiguous()), 512, 256, 512, torch.hann_window(512).pow(0.5)).numpy()
        mm = min(len(x), len(enh)); a, b = x[:mm], enh[:mm]
        sig, bak, ovr, p8 = compute_dnsmos_single(enh, sbo, p808s)
        rows.append({"filename": os.path.basename(cf), "pesq": pesq(SR, a, b, "wb"),
                     "si_sdr": float(si_sdr(a, b)), "estoi": stoi(a, b, SR, extended=True),
                     "dnsmos_sig": sig, "dnsmos_bak": bak, "dnsmos_ovrl": ovr, "p808": p8})
        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/824] PESQ~{np.mean([r['pesq'] for r in rows]):.4f}", flush=True)

    with open("eval_dns/gtcrn_vctk.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    agg = {k: (float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows]))) for k in
           ("pesq", "si_sdr", "estoi", "dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl", "p808")}
    out = {"task": "GTCRN vctk.tar 官方PyTorch, VB-DMD 824(导师裁定: 测, 两值并列)",
           "ckpt": CKPT, "ckpt_sha256": sha, "pipeline": "eval_dns 同管线(48k→16k, 逐文件, 全指标); 官方 STFT(512/256/hann^0.5)模型外; 无归一化(官方流程)",
           "measured_mean_std": agg,
           "published": PUBLISHED,
           "comparison_note": "实测 vs published 并列报差; 若明显低于 published: 排查归一化/重采样/dtype/尺度, 排查不掉如实记未解决差异, 不作 GTCRN 结论(锚点非竞赛)",
           "preregistered": "PLAN §7: 负面答案(1/1361 参数拿大部分质量)可接受=可发表结论, 出数不改口"}
    json.dump(out, open("eval_dns/gtcrn_vctk.json", "w"), indent=2)
    s2 = hashlib.sha256(open("eval_dns/gtcrn_vctk.json", "rb").read()).hexdigest()
    open("eval_dns/gtcrn_vctk.json.sha256", "w").write(f"{s2}  eval_dns/gtcrn_vctk.json\n")
    print(json.dumps({k: round(v[0], 4) for k, v in agg.items()}, indent=1), flush=True)
    print(f"published PESQ(VCTK-DEMAND Table1): {PUBLISHED['pesq']}", flush=True)
    print(f"✓ saved sha={s2[:16]}", flush=True)


if __name__ == "__main__":
    main()
