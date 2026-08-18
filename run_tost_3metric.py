import numpy as np, csv, math
from scipy import stats
def load(p):
    r=list(csv.DictReader(open(p)))
    return {k:np.array([float(x[k]) for x in r]) for k in ('pesq','si_sdr','estoi')}, [x['filename'] for x in r]
m3,nm3=load('eval_dns/m3.csv')
print('=== SI-SDR / ESTOI 配对检验 (⚠️ 只有旧单次抽样 CSV 有这两列, seeded 只存了 pesq) ===')
for tag,path in [('FlowSE N=1','eval_dns/flowse_n1.csv'),('FlowSE N=5','eval_dns/flowse_n5.csv')]:
    f,nf=load(path)
    assert nf==nm3, '文件名顺序不一致!'
    print(f'--- {tag} vs ResFlowSE M3 (文件名逐条对齐 ✓) ---')
    for k,delta in [('pesq',0.05),('si_sdr',0.5),('estoi',0.01)]:
        d=f[k]-m3[k]; n=len(d); mu=d.mean(); se=d.std(ddof=1)/math.sqrt(n)
        t,p=stats.ttest_rel(f[k],m3[k]); lo,hi=stats.t.interval(0.90,n-1,loc=mu,scale=se)
        w=stats.wilcoxon(f[k],m3[k]).pvalue
        verdict='FlowSE 显著更优' if lo>0 else ('FlowSE 显著更差' if hi<0 else 'n.s.')
        eq='等价(δ=%.2f)'%delta if (lo>-delta and hi<delta) else '非等价'
        print('  %-7s Δ=%+8.4f ±%.4f  t_p=%.2e  W_p=%.2e  90%%CI[%+.4f,%+.4f]  %s / %s'%(k,mu,se,p,w,lo,hi,verdict,eq))
