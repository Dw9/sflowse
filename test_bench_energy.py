"""bench_jetson 能耗式 + 恒等式 本地单测(SUBMIT_P0 §怎么验证; 不碰 Jetson)。
验证对象: energy_per_utt(必改1 逐句累加) + identity_checks(必改2 跨口径独立恒等式)。
所有期望值手算, 不依赖模型/设备。"""
import numpy as np
from bench_jetson import energy_per_utt, identity_checks

ATOL = 1e-9
fails = []


def check(name, got, exp, atol=ATOL):
    ok = abs(got - exp) <= atol
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got={got:.6f}  exp={exp:.6f}  |Δ|={abs(got-exp):.2e}")
    if not ok:
        fails.append(name)


print("=== 1. energy_per_utt: 手算期望(必改1 逐句累加)===")
# n=4: P[mW] L[s] D[s], idle=500mW
P = [1000, 1200, 1100, 1300]; L = [0.8, 1.2, 1.0, 1.5]; D = [2.0, 3.0, 2.5, 4.0]; IDLE = 500.0
# Σdur=11.5; Σ(P*L)=800+1440+1100+1950=5290 → E_total=460.0
# Σ((P-idle)*L)=400+840+600+1200=3040 → E_incr=3040/11.5=264.347826...
e = energy_per_utt(P, L, D, IDLE)
check("E_total_mJ_per_s_audio", e["E_total_mJ_per_s_audio"], 5290 / 11.5)
check("E_incr_mJ_per_s_audio", e["E_incr_mJ_per_s_audio"], 3040 / 11.5)
check("audio_total_s", e["audio_total_s"], 11.5)
check("n", e["n"], 4)
print(f"  per_file_E_incr: n={e['per_file_E_incr_mJ_per_s']['n']} "
      f"mean={e['per_file_E_incr_mJ_per_s']['mean']:.4f} (=E_incr {3040/11.5:.4f} 自洽)")

print("\n=== 2. 量纲自检: 恒功率+恒RTF → E_total 应 == P×rtf, E_incr == (P-idle)×rtf ===")
P2 = [1000, 1000, 1000]; L2 = [0.4, 0.4, 0.4]; D2 = [1.0, 1.0, 1.0]
e2 = energy_per_utt(P2, L2, D2, 500.0)
check("恒功率 E_total == P*rtf(=400)", e2["E_total_mJ_per_s_audio"], 1000 * 0.4)
check("恒功率 E_incr == (P-idle)*rtf(=200)", e2["E_incr_mJ_per_s_audio"], 500 * 0.4)

print("\n=== 3. identity_checks: 跨口径独立恒等式(必改2)===")
# wall=5.0s(>Σlat=4.5 → gap=0.5)
ic = identity_checks(L, D, 5.0, P)
check("sum_latency_s", ic["sum_latency_s"], 4.5)
check("lat_over_wall(=0.9)", ic["lat_over_wall"], 0.9)
check("scheduling_gap_s(=0.5)", ic["scheduling_gap_s"], 0.5)
# 路径A 逐句 = 460.0; 路径B 全程 P_mean(=1150)*wall(5)/audio(11.5)=500.0; ratio=0.92
check("E_per_utt_path(=460)", ic["E_per_utt_path"], 460.0)
check("E_global_path(=500)", ic["E_global_path"], 1150.0 * 5.0 / 11.5)
check("E_path_ratio(=0.92)", ic["E_path_ratio"], 460.0 / 500.0)
# Jensen: rtf_pf=[.4,.4,.4,.375] mean=.39375; MoM=4.5/11.5=.391304; ratio>1
check("rtf_per_file_mean(=.39375)", ic["rtf_per_file_mean"], 0.39375)
check("rtf_mean_over_mean(=4.5/11.5)", ic["rtf_mean_over_mean"], 4.5 / 11.5)
check("rtf_jensen_ratio(>1)", ic["rtf_jensen_ratio"], 0.39375 / (4.5 / 11.5))
print(f"  Jensen 方向: per_file_mean({ic['rtf_per_file_mean']:.5f}) > MoM({ic['rtf_mean_over_mean']:.5f})? "
      f"{'✓ 符合预期(长句拉低MoM)' if ic['rtf_per_file_mean'] > ic['rtf_mean_over_mean'] else '✗ 反常'}")

print("\n=== 4. 边界: 空 / 零时长不崩 ===")
print(f"  空: {energy_per_utt([], [], [], 100)}")
print(f"  零时长: {energy_per_utt([100], [1.0], [0.0], 50)}")

print("\n" + ("=" * 50))
print(f"结果: {'全部 PASS ✓' if not fails else f'{len(fails)} FAIL ✗: {fails}'}")
exit(1 if fails else 0)
