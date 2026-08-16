#!/usr/bin/env python3
"""实验 B-2: 幅度域过平滑验证 (跑在 GPU1, 不扰训练)
验证 N=1 复数谱L2最低,但幅度域是否真过平滑(高频能量流失/更平)。
复用 eval_metrics 的加载与 enhance 逻辑; enhance_* 收波形返回 numpy 波形。
"""
import torch, numpy as np, json
from glob import glob
import os
from eval_metrics import (load_audio_16k, enhance_resflowse, enhance_flowse)
from flowmse.resflowse_model import ResFlowSEModel
from flowmse.model import VFModel

SR, NFFT, HOP = 16000, 510, 128
BANDS = [(0,1000),(1000,4000),(4000,8000)]
DATA = '/home/zhibo/workspace/VoiceBank_processed'
WIN = torch.hann_window(NFFT)

def spec_mag(wav_np):
    w = torch.from_numpy(np.asarray(wav_np)).float()
    S = torch.stft(w, NFFT, HOP, window=WIN, return_complex=True)
    return S.abs().numpy()  # [F,T]

def amp_band_mse(mh, mc):
    freqs = np.fft.rfftfreq(NFFT, 1/SR)
    out = {}
    for lo,hi in BANDS:
        m = (freqs>=lo)&(freqs<hi)
        out[f'{lo}-{hi}'] = float(((mh[m]-mc[m])**2).mean())
    out['full'] = float(((mh-mc)**2).mean())
    return out

def hf_retention(mh, mc):
    freqs = np.fft.rfftfreq(NFFT, 1/SR)
    hf = (freqs>=4000)&(freqs<8000)
    eh = (mh**2); ec = (mc**2)
    rh = eh[hf].sum()/(eh.sum()+1e-12)
    rc = ec[hf].sum()/(ec.sum()+1e-12)
    return float(rh/(rc+1e-12))

def flatness(m):
    fs=[]
    for fr in m.T:
        fr = fr[fr>1e-10]
        if len(fr)<2: continue
        fs.append(np.exp(np.log(fr).mean())/(fr.mean()+1e-12))
    return float(np.mean(fs)) if fs else 0.0

def load_flowse(ckpt):
    m = VFModel.load_from_checkpoint(ckpt, base_dir=DATA, map_location='cpu')
    for n,p in m.dnn.named_parameters():
        if n in m.ema_dnn: p.data = m.ema_dnn[n].to(p.device)
    m.eval(); m.cuda(); return m

def load_resflowse(ckpt):
    m = ResFlowSEModel.load_from_checkpoint(ckpt, map_location='cpu', weights_only=False, strict=False)
    m.eval(); m.cuda(); return m  # no_ema: 用原始权重(复现3.062)

if __name__ == '__main__':
    clean = sorted(glob(os.path.join(DATA,'test','clean','*.wav')))
    noisy = sorted(glob(os.path.join(DATA,'test','noisy','*.wav')))
    assert len(clean)==len(noisy)==824, f'{len(clean)}/{len(noisy)}'

    configs = {
        'N=1':      ('flowse', 'VB_DMD_FLOWSE_ICASSP_2025.ckpt', 1),
        'N=5':      ('flowse', 'VB_DMD_FLOWSE_ICASSP_2025.ckpt', 5),
        'Proposed': ('resflowse', 'sflowse.ckpt', None),
    }
    acc = {c:{'amp':[], 'hf':[], 'fl':[]} for c in configs}

    for cname,(kind,ckpt,N) in configs.items():
        print(f'\n=== {cname} ({kind}) ===', flush=True)
        model = load_flowse(ckpt) if kind=='flowse' else load_resflowse(ckpt)
        for i,(cf,nf) in enumerate(zip(clean,noisy)):
            x = load_audio_16k(cf); y = load_audio_16k(nf); T = x.size(1)
            if kind=='flowse':
                xh = enhance_flowse(model, y, T, N=N)
            else:
                xh = enhance_resflowse(model, y, T)
            xn = x.squeeze().numpy()
            L = min(len(xn), len(xh)); xn, xh = xn[:L], xh[:L]
            mh, mc = spec_mag(xh), spec_mag(xn)
            Lm = min(mh.shape[1], mc.shape[1]); mh, mc = mh[:,:Lm], mc[:,:Lm]
            acc[cname]['amp'].append(amp_band_mse(mh, mc))
            acc[cname]['hf'].append(hf_retention(mh, mc))
            acc[cname]['fl'].append(flatness(mh))
            if (i+1)%150==0: print(f'  {i+1}/824', flush=True)
        del model; torch.cuda.empty_cache()

    summary={}
    for c in configs:
        keys = acc[c]['amp'][0].keys()
        summary[c]={
            'amp_mse_bands':{k:float(np.mean([d[k] for d in acc[c]['amp']])) for k in keys},
            'hf_energy_retention':float(np.mean(acc[c]['hf'])),
            'spectral_flatness':float(np.mean(acc[c]['fl'])),
        }
    json.dump({'exp':'B2_amplitude_oversmoothing','n':824,'summary':summary}, open('exp_b2_results.json','w'), indent=2)
    print('\n===== SUMMARY =====')
    for c,d in summary.items():
        print(f"{c}: amp={ {k:round(v,5) for k,v in d['amp_mse_bands'].items()} } hf_ret={d['hf_energy_retention']:.4f} flat={d['spectral_flatness']:.4f}")
