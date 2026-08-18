"""T1 算子级等价性验证 + 假阳性对照 (默认 CUDA 模式运行,不设 NCSNPP_* env).

对照: upfirdn2d()  -> UpFirDn2d.apply  (CUDA op)
       upfirdn2d_native()               (纯 PyTorch, F.pad+F.conv2d)
同进程、同 seed、同输入,直接比 forward+backward max_abs_diff.

假阳性防护: 复制 native 但去掉内部 torch.flip(kernel) 作 "broken_native",
            它与 CUDA 必然差异巨大 -> 判据必须判 FAIL; 否则判据无效, 立刻停.
输出: equiv_fallback_ops.json
"""
import os, json, itertools, torch
import torch.nn.functional as F
import importlib

U = importlib.import_module("flowmse.backbones.ncsnpp_utils.op.upfirdn2d")
upfirdn2d = U.upfirdn2d            # CUDA path (default mode: _USE_NATIVE False)
upfirdn2d_native = U.upfirdn2d_native

DEVICE = "cuda"
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.deterministic = True
torch.manual_seed(0)

SELFATTEST = {
    "mode": "default-CUDA (operator-level reference run)",
    "NCSNPP_PURE_UPFIRDN": os.environ.get("NCSNPP_PURE_UPFIRDN", "0"),
    "NCSNPP_SKIP_CUDA_LOAD": os.environ.get("NCSNPP_SKIP_CUDA_LOAD", "0"),
    "NCSNPP_PURE_PYTORCH": os.environ.get("NCSNPP_PURE_PYTORCH", "0"),
    "upfirdn2d_op_is_None": U.upfirdn2d_op is None,
    "_USE_NATIVE": U._USE_NATIVE,
    "env_set_before_process": True,
    "tf32_disabled": True,
}


def make_input(seed, shape, dtype):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    x = torch.randn(*shape, generator=g, device=DEVICE, dtype=torch.float32)
    k = torch.randn(4, 4, generator=g, device=DEVICE, dtype=torch.float32)  # 4x4 FIR
    return x.to(dtype), k.to(dtype)


def run_case(seed, shape, up, down, pad, dtype):
    x, k = make_input(seed, shape, dtype)
    # CUDA path
    xc = x.detach().clone().requires_grad_(True)
    yc = upfirdn2d(xc, k, up=up, down=down, pad=pad)
    # native path (direct call)
    xn = x.detach().clone().requires_grad_(True)
    yn = upfirdn2d_native(xn, k, up, up, down, down, pad[0], pad[1], pad[0], pad[1])
    assert yc.shape == yn.shape, f"shape mismatch {yc.shape} vs {yn.shape}"
    fwd_mad = (yc.detach() - yn.detach()).abs().max().item()
    grad = torch.randn_like(yc)
    yc.backward(grad)
    yn.backward(grad)
    grad_mad = (xc.grad - xn.grad).abs().max().item()
    return dict(seed=seed, shape=list(shape), up=up, down=down, pad=list(pad),
                dtype=str(dtype), out_shape=list(yc.shape),
                fwd_max_abs_diff=fwd_mad, bwd_max_abs_diff=grad_mad)


# broken native: 复制实现但去掉内部 torch.flip -> 数值必错, 用于证明判据有区分力
def upfirdn2d_native_noflip(input, kernel, up_x, up_y, down_x, down_y, px0, px1, py0, py1):
    _, channel, in_h, in_w = input.shape
    input = input.reshape(-1, in_h, in_w, 1)
    _, in_h, in_w, minor = input.shape
    kh, kw = kernel.shape
    out = input.view(-1, in_h, 1, in_w, 1, minor)
    out = F.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])
    out = out.view(-1, in_h * up_y, in_w * up_x, minor)
    out = F.pad(out, [0, 0, max(px0, 0), max(px1, 0), max(py0, 0), max(py1, 0)])
    out = out[:, max(-py0, 0): out.shape[1] - max(-py1, 0), max(-px0, 0): out.shape[2] - max(-px1, 0), :]
    out = out.permute(0, 3, 1, 2).reshape(-1, 1, in_h * up_y + py0 + py1, in_w * up_x + px0 + px1)
    w = kernel.view(1, 1, kh, kw)                       # <<< 故意去掉 torch.flip (破坏)
    out = F.conv2d(out, w)
    out = out.reshape(-1, minor, in_h * up_y + py0 + py1 - kh + 1, in_w * up_x + px0 + px1 - kw + 1)
    out = out.permute(0, 2, 3, 1)[:, ::down_y, ::down_x, :]
    oh = (in_h * up_y + py0 + py1 - kh) // down_y + 1
    ow = (in_w * up_x + px0 + px1 - kw) // down_x + 1
    return out.view(-1, channel, oh, ow)


def run_broken(seed, shape, up, down, pad):
    x, k = make_input(seed, shape, torch.float32)
    yc = upfirdn2d(x, k, up=up, down=down, pad=pad).detach()
    yb = upfirdn2d_native_noflip(x, k, up, up, down, down, pad[0], pad[1], pad[0], pad[1]).detach()
    assert yc.shape == yb.shape
    return (yc - yb).abs().max().item()


SHAPES = [(2, 4, 16, 16), (1, 8, 32, 32), (2, 3, 20, 24), (3, 2, 48, 48),
          (1, 1, 12, 12), (2, 6, 28, 28)]
PARAMS = [(1, 1, (0, 0)), (1, 1, (1, 1)), (1, 1, (2, 2)),
          (2, 1, (1, 1)), (2, 2, (3, 3)), (1, 2, (2, 2))]

print(f"path self-attest: op_is_None={SELFATTEST['upfirdn2d_op_is_None']} _USE_NATIVE={SELFATTEST['_USE_NATIVE']}")

results_fp32, results_fp16 = [], []
cuda_fp16_ok = True
for i, (shape, (up, down, pad)) in enumerate(itertools.product(SHAPES, PARAMS)):
    seed = 1000 + i
    results_fp32.append(run_case(seed, shape, up, down, pad, torch.float32))
    try:
        results_fp16.append(run_case(seed, shape, up, down, pad, torch.float16))
    except Exception as e:
        cuda_fp16_ok = False
        results_fp16.append(dict(seed=seed, shape=list(shape), up=up, down=down, pad=list(pad),
                                 dtype="float16", error=str(e)[:200]))

# 假阳性对照
broken = run_broken(seed=9999, shape=(2, 4, 32, 32), up=2, down=1, pad=(1, 1))
FP32_THRESH = 1e-5
sanity_judge_FAIL = broken >= FP32_THRESH   # 必须为 True

# fp32 硬门
fp32_fwd_pass = all(c["fwd_max_abs_diff"] < FP32_THRESH for c in results_fp32)
fp32_bwd_pass = all(c["bwd_max_abs_diff"] < FP32_THRESH for c in results_fp32)

# fp16 噪声底对照 (只报不 gate): |native_fp16 - cuda_fp16| vs |cuda_fp16 - cuda_fp32.half()|
fp16_noise = None
if cuda_fp16_ok:
    pairs = []
    for c16, c32 in zip(results_fp16, results_fp32):
        if "error" in c16:
            continue
        # 用同一 seed 重算三份干净对照
        x, k = make_input(c16["seed"], tuple(c16["shape"]), torch.float32)
        y_cuda_fp32 = upfirdn2d(x, k, up=c16["up"], down=c16["down"], pad=tuple(c16["pad"])).detach()
        x16 = x.half(); k16 = k.half()
        y_cuda_fp16 = upfirdn2d(x16, k16, up=c16["up"], down=c16["down"], pad=tuple(c16["pad"])).detach()
        y_native_fp16 = upfirdn2d_native(x16, k16, c16["up"], c16["up"], c16["down"], c16["down"],
                                         c16["pad"][0], c16["pad"][1], c16["pad"][0], c16["pad"][1]).detach()
        intro = (y_native_fp16.float() - y_cuda_fp16.float()).abs().max().item()
        noise = (y_cuda_fp16.float() - y_cuda_fp32.half().float()).abs().max().item()
        pairs.append(dict(seed=c16["seed"], native_minus_cuda_fp16=intro, cuda_fp16_minus_fp32half=noise))
    fp16_noise = pairs

out = {
    "path_selfattest": SELFATTEST,
    "fp32": {
        "n_cases": len(results_fp32),
        "fwd_max_abs_diff_max": max(c["fwd_max_abs_diff"] for c in results_fp32),
        "bwd_max_abs_diff_max": max(c["bwd_max_abs_diff"] for c in results_fp32),
        "fwd_pass_lt_1e-5": fp32_fwd_pass,
        "bwd_pass_lt_1e-5": fp32_bwd_pass,
        "cases": results_fp32,
    },
    "fp16": {
        "cuda_fp16_supported": cuda_fp16_ok,
        "n_cases": len(results_fp16),
        "note": "只报不 gate; 判据=回退引入误差 |native-cuda|_fp16 不大于 fp16 量化本身噪声底 |cuda_fp16-cuda_fp32.half| 同量级",
        "noise_floor_pairs": fp16_noise,
    },
    "sanity_broken_native": {
        "fwd_max_abs_diff": broken,
        "threshold": FP32_THRESH,
        "judge_FAIL_expected_True": sanity_judge_FAIL,
    },
    "meta": {"torch": torch.__version__, "device": torch.cuda.get_device_name(0)},
}

with open("equiv_fallback_ops.json", "w") as fp:
    json.dump(out, fp, indent=2)

print(json.dumps({k: out[k] for k in ["path_selfattest", "fp32", "sanity_broken_native", "meta"]}, indent=2))
print("fp16 noise floor pairs (first 3):", json.dumps((fp16_noise or [])[:3], indent=2))
print(f"\n>>> 写入 equiv_fallback_ops.json")
print(f">>> 假阳性 broken fwd_max_abs_diff={broken:.3e} (判据 FAIL={sanity_judge_FAIL}) -- {'OK 有区分力' if sanity_judge_FAIL else '!!! 无区分力, 立刻停 !!!'}")
print(f">>> fp32 fwd_pass={fp32_fwd_pass} bwd_pass={fp32_bwd_pass}")
