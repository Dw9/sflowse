"""prep_descod_local.py — P6 DeScoD-ECG 前置验证(导师四条: 只读 third_party / 独立 .venv_descod /
随机初始化实例化+前向+cuda.Event 计时管线验证 / 不报数字入稿; 步数轴调查=关键)。
note 写死: '仅代价侧框架迁移; N* 与决策规则不迁移'。
产出: prep_descod_local.json + 旁挂 sha(仓库根路径)。
"""
import os, sys, json, hashlib, time
import numpy as np
import torch

REPO = "/home/zhibo/workspace/sflowse/third_party/descod-ecg"
sys.path.insert(0, REPO)
# LEAF frontend 是残留 import(代码中仅出现在注释行), 模型不用 → 桩掉避免装整个 leaf 依赖
import types
_leaf = types.ModuleType("leaf_audio_pytorch"); _fe = types.ModuleType("leaf_audio_pytorch.frontend")
import importlib.machinery as im
_leaf.__spec__ = im.ModuleSpec("leaf_audio_pytorch", None); _fe.__spec__ = im.ModuleSpec("leaf_audio_pytorch.frontend", None)
sys.modules["leaf_audio_pytorch"] = _leaf; sys.modules["leaf_audio_pytorch.frontend"] = _fe

import yaml
from denoising_model_small import ConditionalModel
from main_model import DDPM

dev = "cuda"
cfg = yaml.safe_load(open(os.path.join(REPO, "config/base.yaml")))
n_steps_cfg = cfg["diffusion"]["num_steps"]

out = {"task": "P6 前置: DeScoD-ECG 随机初始化实例化+前向+cuda.Event 计时管线验证(A100)",
       "note": "仅代价侧框架迁移; N* 与决策规则不迁移; 不报数字入稿(此处数字仅管线验证用)",
       "repo": "HuayuLiArizona/Score-based-ECG-Denoising", "commit": "f32cd5a (2025-11-17)",
       "venv": ".venv_descod (torch 2.13.0+cu130, 独立, third_party 只读)"}

# ① 实例化(随机初始化, 不加载其 checkpoint)
torch.manual_seed(0)
model = ConditionalModel(feats=cfg["train"]["feats"]).to(dev)
n_params = sum(p.numel() for p in model.parameters())
out["n_params"] = n_params
print(f"[1] ConditionalModel 随机初始化 OK, params={n_params:,}", flush=True)

# ② 前向(条件去噪分支 shape 探测: ECG 400Hz, 输入 (B,1,L))
ddpm = DDPM(model, cfg, dev)
for L in (4000, 8000):
    x = torch.randn(1, 1, L, device=dev)
    with torch.no_grad():
        y = model(x, x, torch.tensor([[0.5]], device=dev))
    out.setdefault("forward_shapes", []).append({"input": list(x.shape), "output": list(y.shape)})
    print(f"[2] forward L={L}: {list(x.shape)} -> {list(y.shape)}", flush=True)

# ③ 步数轴: 改 config num_steps → DDPM 重置 beta schedule(采样循环 num_steps 次)
axis = {}
for ns in (10, 25, 50, 100):
    cfg2 = yaml.safe_load(open(os.path.join(REPO, "config/base.yaml")))
    cfg2["diffusion"]["num_steps"] = ns
    d2 = DDPM(model, cfg2, dev)
    ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
    x = torch.randn(1, 1, 4000, device=dev)
    with torch.no_grad():
        d2.denoising(x, continous=False)          # warmup(含JIT)
        ev0.record(); t0 = time.time()
        d2.denoising(x, continous=False)
        ev1.record(); torch.cuda.synchronize()
    axis[str(ns)] = {"cuda_ms": round(ev0.elapsed_time(ev1), 2), "wall_ms": round((time.time()-t0)*1000, 2)}
    print(f"[3] num_steps={ns}: denoising cuda {axis[str(ns)]['cuda_ms']}ms", flush=True)
out["step_axis"] = axis
out["step_axis_verdict"] = ("AXIS EXISTS: num_steps 在 config['diffusion'] 控制 beta schedule, 采样循环 "
    "main_model.py p_sample_loop: `for i in reversed(range(0, self.num_steps))` — 每步一次 model() 调用; "
    "框架可迁移(代价侧 N_max_p95 可在 Jetson 计算)")
out["evidence_lines"] = {"p_sample_loop": "main_model.py L134-141: for i in reversed(range(0, self.num_steps)): cur_x = self.p_sample(cur_x, i, condition_x=x)",
                          "num_steps_config": "config/base.yaml diffusion.num_steps = 50 (默认)"}

json.dump(out, open("prep_descod_local.json", "w"), indent=2, ensure_ascii=False)
sha = hashlib.sha256(open("prep_descod_local.json", "rb").read()).hexdigest()
open("prep_descod_local.json.sha256", "w").write(f"{sha}  prep_descod_local.json\n")
print(f"\n✓ saved prep_descod_local.json sha={sha[:16]}", flush=True)
print(f"  步数轴: {'有' if axis else '?'} | 参数量 {n_params:,} | 结果不入稿", flush=True)
