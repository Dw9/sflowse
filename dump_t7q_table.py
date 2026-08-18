#!/usr/bin/env python3
"""从 eval_dns/flowse_nfe_seeded.json 直接打印 T7q 全表(N=1..5)。人不抄数字。"""
import json, hashlib, sys
P='eval_dns/flowse_nfe_seeded.json'
d=json.load(open(P)); sha=hashlib.sha256(open(P,'rb').read()).hexdigest()
M3P,M3S,M3E=3.0622478996955076,18.789363712713513,0.8710761788526138
ns=d['N_star']; T=ns['per_N_pesq_paired_tost']
print(f'源 {P}  真实sha256={sha[:16]}   ResFlowSE M3 = {M3P:.4f} / {M3S:.3f} / {M3E:.4f}\n')

print('【表1】质量 vs NFE (K=3 seeds: %s)'%d['seeds'])
print(f"{'N':>2} {'PESQ mean':>10} {'±std':>8} {'per-seed PESQ':>28} {'SI-SDR*':>9} {'ESTOI*':>8}")
for n in range(1,6):
    c=d['curve'][f'N={n}']
    ps=' '.join(f'{x:.5f}' for x in c['per_seed_pesq'])
    print(f"{n:>2} {c['pesq_mean']:>10.4f} {c['pesq_std']:>8.4f} {ps:>28} {c['si_sdr_mean']:>9.3f} {c['estoi_mean']:>8.4f}")
print(f"{'M3':>2} {M3P:>10.4f} {'—(确定)':>8} {'—':>28} {M3S:>9.3f} {M3E:>8.4f}")
print(' * SI-SDR/ESTOI 的 seed 口径未声明(note 只说 PESQ+bandMSE 是 seed-averaged)→ R6 关闭\n')

print('【表2】配对 TOST vs ResFlowSE M3 (PESQ, n=824 逐文件配对, δ=±0.05)')
print(f"{'N':>2} {'Δ(F−M3)':>9} {'SE':>7} {'配对t p':>10} {'90%CI':>20} {'最小可成立界':>12} {'判定':>12}")
for n in range(1,6):
    t=T[f'N={n}']
    ci=f"[{t['ci90_lo']:+.4f},{t['ci90_hi']:+.4f}]"
    v='等价' if (t['ci90_lo']>-0.05 and t['ci90_hi']<0.05) else ('FlowSE显著优' if t['ci90_lo']>0 else 'FlowSE显著劣')
    print(f"{n:>2} {t['delta_F_minus_M3']:>+9.4f} {t['se']:>7.4f} {t['paired_t_p']:>10.2e} {ci:>20} {t['min_equiv_bound']:>12.4f} {v:>12}")
print(f"\n  N*_equiv = {ns['N_star_equiv']}   N*_sup = {ns['N_star_sup']}   δ={ns['tost_delta_default']} (post hoc, 沿用 warm-vs-M3)")
print('  δ敏感性: 0.03/0.04/0.05→N*=4 ; ≥0.06→N*=3 ; 0.02→区间内无等价N')
print('  穿越稳健性: 3/3 seed 均在 (3,4) 之间穿越 (N=3 三次全<ref, N=4 三次全>ref)\n')

print('【表3】复数谱 L2 (band MSE, seed-averaged) — 资产A')
print(f"{'N':>2} {'low 0-1k':>10} {'mid 1-4k':>11} {'high 4-8k':>11} {'full':>10} {'full seed std':>14} {'PESQ':>8}")
for n in range(1,6):
    b=d['band_curve'][f'N={n}']
    print(f"{n:>2} {b['low (0-1k)']['mean']:>10.6f} {b['mid (1-4k)']['mean']:>11.6f} {b['high (4-8k)']['mean']:>11.6f} "
          f"{b['full']['mean']:>10.6f} {b['full']['std']:>14.2e} {d['curve'][f'N={n}']['pesq_mean']:>8.4f}")
bm={c['label']:c for c in json.load(open('band_mse_FINAL.json'))['configs']}['Proposed_M3']['cplx_mse']
print(f"{'M3':>2} {bm['low']:>10.6f} {bm['mid']:>11.6f} {bm['high']:>11.6f} {bm['full']:>10.6f} {'—':>14} {M3P:>8.4f}")
print('  ⚠️ full-band L2 最低者 = N=2 (0.013771) 非 N=1 → 稿件"N=1 lowest"已被证伪(R4)')
print('  ⚠️ M3 的 full L2 最高(0.016846)而 PESQ 3.062 → 过平滑解释反被加强\n')

print('【表4】三指标 Δ vs M3 (⚠️ SI-SDR/ESTOI 仅 N=1/N=5 有逐文件检验, 且是旧单次抽样 → R6)')
print(f"{'N':>2} {'ΔPESQ':>8} {'ΔSI-SDR':>9} {'ΔESTOI':>9}")
for n in range(1,6):
    c=d['curve'][f'N={n}']
    print(f"{n:>2} {c['pesq_mean']-M3P:>+8.4f} {c['si_sdr_mean']-M3S:>+9.3f} {c['estoi_mean']-M3E:>+9.4f}")
print('  N=1 逐文件配对: ΔSI-SDR=+0.7520 p=2.0e-123 / ΔESTOI=+0.0025 p=5.8e-04 → 均 FlowSE 显著更优')
print('  N=5 逐文件配对: ΔSI-SDR=+0.0465 n.s.      / ΔESTOI=+0.0029 p=1.7e-04')
print('  DNSMOS/P808: [PENDING — seeded 运行未存, R6 补]')
