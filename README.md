# Solver Steps or Solver-Free Retraining? — Measurement Study Code

Code for a same-board latency–energy–quality characterization of flow-matching speech
enhancement on a power-constrained embedded GPU (Jetson Orin NX). This repository contains
the **model code, training, evaluation, statistical analysis, and the on-device measurement
protocol** used in the paper:

> *Solver Steps or Solver-Free Retraining? A Same-Board Latency–Energy–Quality
> Characterization of Flow-Matching Speech Enhancement on a Power-Constrained Embedded GPU*

The paper is a **measurement and characterization study** — no new enhancement method is
proposed. The code here exists to make every reported number reproducible and auditable.

## What is measured

Two quantities, both on the same hardware, combine into a deployment decision rule:

- **N\*** (quality side, measured on GPU server): the smallest solver-step count at which the
  truncated generative model (FlowSE) is statistically equivalent (TOST) to a single-step
  retrained model (ResFlowSE) — PESQ caliber, N\* = 4.
- **N_max** (cost side, measured on Jetson Orin NX across four power modes): the largest
  solver-step count that stays under RTF = 1, reported for both the per-file mean and p95
  criteria (MAXN: 6/5; 25 W / 15 W / 10 W: 2/2 each).

Conditional decision rule (on this corpus, among the compared recipes): where N_max ≥ N\*,
keeping the solver is feasible and preferred; where N_max < N\*, only retraining reaches the
retrained model's PESQ quality. On this board, every constrained power budget selects retraining.

## Model checkpoints

| Checkpoint | Role | SHA-256 | Where |
|---|---|---|---|
| FlowSE teacher (VB-DMD) | Released generative teacher; truncation runs and warm-start initialization | `9f4502703ccb9b252135884f1d72cb64bc6dbd23d0f23dc8b01c1b2f07bf329c` | [Google Drive — VB-DMD checkpoint](https://drive.google.com/file/d/1PgzFSAu2t3BX8znNuPKNZdvaOSEXeo88/view) (from the [upstream FlowSE repository](https://github.com/seongq/flowmse); other upstream checkpoints are listed there) |
| ResFlowSE (M3, `sflowse.ckpt`) | Single-step retrained model — the paper's main model | `7c3e3a3e3f65028bb47a856f8a56f78dc81860419df7283cec7bf51f543d0196` | released upon paper acceptance |
| cold @epoch 24 | Initialization-ablation baseline (random init) | `44eb87bdc91fccf5de6d473c94ce347f951300687e7b17d1e47cdf9246625c58` | released upon paper acceptance |
| warm @epoch 36 | Initialization ablation (FlowSE-weights init) | `efeacbce4cca239aa6682b2a08e383de7d5c19cce5865efbaa2c9428ab86b5c4` | released upon paper acceptance |

Checkpoints are selected by best validation PESQ on a 200-file training-side validation subset
(cold: epoch 24; warm: epoch 36; M3: released epoch 39 — the end of a staged multi-run training
schedule; see the paper's §5 for the full training graphs).

## Repository map

| Path | What it is |
|---|---|
| `flowmse/` | Model package: NCSN++ backbone, flow-matching / residual single-step models |
| `train.py`, `train_resflowse.py` | Training: original flow-matching model; single-step retrained variant |
| `eval_metrics.py`, `evaluate.py` | Full-test-set quality evaluation (PESQ / SI-SDR / ESTOI / DNSMOS) |
| `eval_flowse_nfe_seeded_v2.py` | Quality-vs-solver-steps curve, K=3 seeds (the N\* source) |
| `gtcrn_quality_vb.py` | Cross-family anchor (GTCRN) quality, same corpus and pipeline |
| `run_sigtest.py`, `run_tost_nstar_v2.py`, `run_tost_7metric.py` | Paired significance tests and TOST equivalence (N\* derivation; seven-metric equal-cost picture) |
| `run_band_mse.py`, `run_exp_b2.py`, `run_2ch_equiv.py` | Band-wise spectral analysis; two-channel equivalence check |
| `bench_t2a.py`, `a100_toaudio_bench.py` | GPU-server latency benchmarks; iSTFT direct timing |
| `bench_jetson.py` | **On-device measurement protocol** (four power modes; calibration, identity and linearity gates; per-utterance energy accumulation; environment snapshots recorded inside every artifact) |
| `bench_gtcrn_jetson.py` | Same-protocol anchor benchmark (GTCRN) on the same board |
| `maxn_ab_interleaved.py`, `p2_ab_interleaved.py`, `maxn_n6_confirm.py`, `sustained_bench.py` | Clean-environment cross-checks; full-corpus boundary confirmation; sustained-caliber energy |
| `p4_precheck.py` | Power-telemetry sampling-validity precheck (falsification gate for per-utterance energy) |
| `onnx_smoke.py` | Export-path probing (complex-tensor boundary) |
| `p7_chunk_quality.py` | Chunked-inference quality cost (offline feasibility boundary) |
| `nfe_energy_n1to5.py`, `n56_perfile.py` | Per-N energy on-device pass (N=1–5) and MAXN N=5/6 per-file RTF arrays (p95 bootstrap source) |
| `analyze_firstfile_p95.py` | First-retained-file sensitivity analysis |
| `make_figures.py`, `make_tables.py` | Paper figures and tables, generated **directly from measurement artifacts** (no hand-entered numbers) |
| `check_numbers.py` | Number-traceability guard: every manuscript number must trace to the data ledger; `--selftest` verifies the guard fails on known-bad inputs (fail-closed) |
| `check_artifacts.py` | Artifact guard: SHA-256 sidecar integrity over all measurement JSONs (plus SRC-cited NPYs) and unique-source checks on the thermal drift values; fail-closed with `--selftest` |
| `check_reforder.py`, `check_headings.py` | Reference first-appearance order; heading/structure guards |

## Reproducibility discipline

- **Every manuscript number is generated from a measurement artifact** (JSON with an
  sidecar SHA-256), never hand-entered. `make_tables.py` / `make_figures.py` read the
  artifacts directly; `check_numbers.py` re-verifies the manuscript against the ledger and
  exits non-zero on any untraceable or retracted number.
- **On-device artifacts are self-describing**: power-mode confirmation, environment/process
  snapshots, sampling interval, and per-utterance records with file identifiers are recorded
  inside each artifact, so entry conditions need no archaeology.
- Measurement artifacts themselves (JSON/CSV/NPY) and the internal audit trail are not
  committed to this repository; the paper reports them with per-artifact hashes.

## Environment

- Python 3.10, PyTorch 2.x + Lightning (GPU server: A100; device: Jetson Orin NX,
  JetPack R36.x, torch aarch64 build — **do not replace the device torch with a wheels build**).
- VoiceBank-DEMAND test set (824 utterances), 48 kHz stored / 16 kHz processed.
- Third-party model code (GTCRN) is used read-only from its official repository at a pinned
  commit; it is not vendored here.

## Usage sketch

```bash
# quality: K-seed solver-step curve + equivalence tests
python eval_flowse_nfe_seeded_v2.py --seeds 3 --nfe 1..5
python run_tost_nstar_v2.py

# device: four power modes, calibration + gates + energy
sudo nvpmodel -m <mode> && sudo jetson_clocks   # confirm with nvpmodel -q
python bench_jetson.py --mode_id <m> --n_real 824 --subset_n 100 --calib_full824 \
    --nfe_list 1,2,3,4,5,6 --idle_cold_s 120 --idle_hot_s 60

# regenerate paper assets from artifacts; verify number traceability
python make_figures.py && python make_tables.py
python check_numbers.py --selftest && python check_numbers.py
```
