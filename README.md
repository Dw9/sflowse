# ResFlowSE: Single-Step Speech Enhancement via Flow Matching


## Overview

ResFlowSE achieves **comparable performance** to multi-step flow matching models (FlowSE) while being **4.13× faster** through:
- **Residual learning**: predicting the clean-noisy difference instead of absolute clean speech
- **Energy-adaptive conditioning**: modulating the backbone network based on noisy speech energy
- **Multi-resolution loss**: combining magnitude, complex, multi-resolution STFT, and SI-SDR losses

## Performance (VoiceBank-DEMAND)

| Model | PESQ ↑ | SI-SDR (dB) ↑ | ESTOI ↑ | RTF ↓ |
|-------|--------|---------------|---------|-------|
| FlowSE (5-NFE) | 3.089±0.686 | 18.85±3.18 | 0.874±0.095 | 0.081 |
| **ResFlowSE (1-NFE)** | **3.062±0.638** | **18.789±3.299** | **0.871±0.097** | **0.020** |

*measured on RTX 4090 GPU*

## Installation

```bash
# Create virtual environment
uv venv --python 3.10
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

## Dataset Preparation

Organize your dataset as follows:
```
dataset_dir/
├── train/
│   ├── clean/
│   └── noisy/
├── valid/
│   ├── clean/
│   └── noisy/
└── test/
    ├── clean/
    └── noisy/
```

Each `clean/` and `noisy/` subdirectory should contain matching `.wav` files.

## Training

### Train ResFlowSE (single-step)
```bash
python train_resflowse.py --base_dir <dataset_dir> --backbone ncsnpp
```

### Train FlowSE (multi-step baseline)
```bash
python train.py --base_dir <dataset_dir> --backbone ncsnpp
```

### Options
- `--backbone`: Choose backbone architecture (`ncsnpp` or `dcunet`)
- `--no_wandb`: Disable Weights & Biases logging (uses TensorBoard instead)
- `--ckpt`: Resume from checkpoint

## Evaluation

### checkpoint 

our model on voicebank-demand: [Download](https://drive.google.com/file/d/1MUCVtKRAR0X9EXhPvBb7yvGcygEItZN2/view?usp=sharing)

flowse model on voicebank-demand: [Download](https://drive.google.com/file/d/1PgzFSAu2t3BX8znNuPKNZdvaOSEXeo88/view?usp=drive_link)

### ResFlowSE (single-step)
```bash
python eval_metrics.py \
  --model_type resflowse \
  --ckpt <checkpoint_path> \
  --data_dir <dataset_dir> \
  --no_ema \
  --split test
```

### FlowSE (multi-step baseline)
```bash
python eval_metrics.py \
  --model_type flowse \
  --ckpt <checkpoint_path> \
  --data_dir <dataset_dir> \
  --N 5 \
  --split test 
```

### Options
- `--split`: Dataset split to evaluate (`test` by default)
- `--N`: Number of ODE steps for FlowSE (ignored for ResFlowSE)
- `--no_ema`: Skip EMA weight swap (required for ResFlowSE)
- `--output_csv <path>`: Save per-utterance results to CSV
- `--num_files <n>`: Limit number of files for quick testing

### Speed Benchmark
```bash
python benchmark_speed.py \
  --resflowse_ckpt <path> \
  --flowse_ckpt <path> \
  --data_dir <dataset_dir>
```

## Acknowledgments

This work builds upon [FlowSE](https://github.com/sp-uhh/flowse) (ICASSP 2025) and the [SGMSE+](https://github.com/sp-uhh/sgmse) codebase.

## License

MIT License
