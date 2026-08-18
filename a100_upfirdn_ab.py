"""a100_upfirdn_ab.py — 算子级 A/B(导师决定性实验, 判读预写死)
同一随机张量过 upfirdn2d_native vs CUDA upfirdn2d_op, 用模型实际调用形态(up/down/pad, conv 后调用)。
判读: (a)近似相同→上游等价, 塌陷另有原因, 归因作废重查; (b)显著不同→上游 native 与 CUDA kernel 不数值等价(可写的工程发现); (c)SKIP_CUDA_LOAD-only 也塌→我们的 patch 有问题。
"""
import os, sys, json, hashlib
import torch

# 进程必须先看两种模式: 用子进程分别跑 native 与 CUDA
MODE = sys.argv[1] if len(sys.argv) > 1 else None
OUT = {}

def run():
    import importlib
    mod = importlib.import_module("flowmse.backbones.ncsnpp_utils.op.upfirdn2d")
    importlib.reload(mod)
    torch.manual_seed(42)
    dev = "cuda"
    res = {}
    # 模型实际用的形态(up_or_down_sampling.py): [N,C,H,W] float32, k=setup_kernel, up/down factor=2, 各种 pad
    from flowmse.backbones.ncsnpp_utils.up_or_down_sampling import _setup_kernel
    cases = {
        "upsample_factor2": dict(up=2, k=[1,3,3,1], gain=4, x=(1,4,64,80)),
        "downsample_factor2": dict(down=2, k=[1,3,3,1], gain=1, x=(1,4,64,80)),
        "up_then_conv": dict(up=2, k=[1,3,3,1], gain=1, x=(1,8,32,40)),
    }
    for name, c in cases.items():
        k = torch.as_tensor(_setup_kernel(c["k"]) * c.get("gain", 1.0), dtype=torch.float32, device=dev)
        if c.get("up"): 
            p = k.shape[0] - c["up"]; pad = ((p+1)//2 + c["up"] - 1, p//2)
        else:
            p = k.shape[0] - c["down"]; pad = ((p+1)//2, p//2)
        x = torch.randn(*c["x"], device=dev)
        if c.get("up"): kw = dict(up=c["up"], pad=pad)
        else: kw = dict(down=c["down"], pad=pad)
        y = mod.upfirdn2d(x, k, **kw)
        res[name] = {"shape": list(y.shape), "checksum": float(y.double().sum()), "float_sum": float(y.sum()), "max_abs": float(y.abs().max())}
        torch.save(y.cpu(), f"ab_{MODE}_{name}.pt")
    return res

if MODE:
    OUT["mode"] = MODE
    OUT["env"] = {kk: os.environ.get(kk) for kk in ("NCSNPP_PURE_PYTORCH","NCSNPP_PURE_UPFIRDN","NCSNPP_SKIP_CUDA_LOAD")}
    OUT["op_path"] = "native" if (os.environ.get("NCSNPP_PURE_UPFIRDN")=="1" or os.environ.get("NCSNPP_PURE_PYTORCH")=="1") else "cuda_op"
    OUT["results"] = run()
    print(json.dumps(OUT, indent=1), flush=True)
    sys.exit(0)

# 调度: 两个独立进程(native=pure env / cudaop=默认编译CUDA)
import subprocess
os.environ["PATH"] = os.path.abspath(".venv/bin") + ":" + os.environ["PATH"]
env_n = dict(os.environ); env_n["NCSNPP_PURE_PYTORCH"] = "1"  # master: upfirdn native + fused_act 也跳过编译(ninja 缺)
env_c = dict(os.environ);  # 默认: 编译并使用 CUDA op
for tag, env in (("native", env_n), ("cudaop", env_c)):
    r = subprocess.run([sys.executable, __file__, tag], capture_output=True, text=True, env=env)
    print(r.stdout[-400:], r.stderr[-200:] if r.returncode else "", flush=True)

# 盘上比对(判读预写死)
out = {"task": "upfirdn2d native vs CUDA op 算子级 A/B(导师决定性实验)"}
for name in ("upsample_factor2", "downsample_factor2", "up_then_conv"):
    a = torch.load(f"ab_native_{name}.pt").double()
    b = torch.load(f"ab_cudaop_{name}.pt").double()
    if a.shape != b.shape:
        out[name] = {"verdict": "SHAPE MISMATCH", "native": list(a.shape), "cuda": list(b.shape)}; continue
    d = (a - b).abs()
    rel = d.mean() / (a.abs().mean() + 1e-12) * 100
    v = "BITWISE" if d.max() == 0 else ("EQUIVALENT" if rel < 1e-6 else ("DIFFERENT"))
    out[name] = {"verdict": v, "max_abs_delta": float(d.max()), "rel_err_pct": float(rel),
                 "native_maxabs": float(a.abs().max()), "cuda_maxabs": float(b.abs().max())}
    print(name, out[name], flush=True)

out["reading_prewritten"] = {"EQUIVALENT/BITWISE": "(a) 上游等价 → 塌陷另有原因, 归因作废重查",
                              "DIFFERENT": "(b) 上游 native 与 CUDA kernel 不数值等价 → 可写的工程发现",
                              "SKIP_CUDA_LOAD-only也塌": "(c) 我们的 patch 有问题, 是我们的锅"}
json.dump(out, open("a100_upfirdn_ab.json", "w"), indent=2)
sha = hashlib.sha256(open("a100_upfirdn_ab.json", "rb").read()).hexdigest()
open("a100_upfirdn_ab.json.sha256", "w").write(f"{sha}  a100_upfirdn_ab.json\n")
print(f"✓ saved a100_upfirdn_ab.json sha={sha[:16]}", flush=True)
