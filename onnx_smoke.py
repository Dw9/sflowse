"""T1 ONNX smoke test (补充3 三级验收). PURE_PYTORCH 模式运行.
三级: (i) torch.onnx.export 成功? (legacy + dynamo) (ii) onnxruntime 加载? (iii) ORT vs PyTorch max_abs_diff?
导出目标 = ResFlowSEModel.forward(complex 谱 [B,1,F,T]). complex 是预期拦路虎, 失败如实记录.
"""
import os, json, traceback, torch
from flowmse.resflowse_model import ResFlowSEModel

OPSET = 17
m = ResFlowSEModel.load_from_checkpoint("sflowse.ckpt", map_location="cpu",
                                        weights_only=False, strict=False)
m.cuda().eval()

torch.manual_seed(42)
Y = torch.randn(1, 1, 256, 256, device="cuda", dtype=torch.complex64)
with torch.no_grad():
    PT = m.forward(Y)
results = {"opset": OPSET, "input": {"shape": list(Y.shape), "dtype": str(Y.dtype)},
           "pt_out_dtype": str(PT.dtype)}


def _tail(msg, n=900):
    return (msg[-n:] if len(msg) > n else msg)


# ---------- (i) export ----------
for tag, kw in [("legacy", {"dynamo": False, "verbose": False}), ("dynamo", {"dynamo": True})]:
    path = f"resflowse_t1_{tag}.onnx"
    rec = {"path": path}
    try:
        torch.onnx.export(m, (Y,), path, opset_version=OPSET,
                          input_names=["y"], output_names=["x_hat"], **kw)
        rec["export_success"] = True
        rec["size_bytes"] = os.path.getsize(path) if os.path.exists(path) else 0
    except Exception as e:
        rec["export_success"] = False
        rec["error_type"] = type(e).__name__
        rec["error_tail"] = _tail(str(e))
        rec["traceback_tail"] = _tail(traceback.format_exc())
    results[tag] = rec


# ---------- (ii)+(iii) onnxruntime load + 数值比对 ----------
def _try_ort(tag):
    path = f"resflowse_t1_{tag}.onnx"
    rec = {"path": path, "exists": os.path.exists(path)}
    if not rec["exists"]:
        rec["ort_load_success"] = False
        rec["note"] = "onnx 文件未生成(export 失败)"
        return rec
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        rec["ort_load_success"] = True
        rec["inputs"] = [{"name": i.name, "type": str(i.type), "shape": i.shape} for i in sess.get_inputs()]
        rec["outputs"] = [{"name": o.name, "type": str(o.type), "shape": o.shape} for o in sess.get_outputs()]
        # (iii) 数值比对: complex 输入 ONNX 表示可能拆 real/imag. 尝试直接喂 complex np(失败再试拆分).
        import numpy as np
        y_np = Y.detach().cpu().numpy()
        inp0 = sess.get_inputs()[0]
        fed = False
        # 尝试 A: 直接喂 complex np array
        try:
            ort_out = sess.run(None, {inp0.name: y_np})
            fed = True
        except Exception as eA:
            rec["feed_complex_direct_error"] = _tail(str(eA))
            # 尝试 B: 若 input shape 最后维 == 2*complex_last(实虚拼接)
            try:
                y_view = torch.view_as_real(Y).float().cpu().numpy()  # [..., 2]
                ort_out = sess.run(None, {inp0.name: y_view})
                fed = True
            except Exception as eB:
                rec["feed_real_imag_view_error"] = _tail(str(eB))
        if fed:
            pt_np = PT.detach().cpu().numpy()
            o = ort_out[0]
            # 对齐: ort 输出可能是 real [...,2] 而 pt 是 complex
            try:
                diff = np.abs(o.astype(complex) if o.dtype != np.complex64 else o) - pt_np
                rec["ort_vs_pt_max_abs_diff"] = float(np.max(np.abs(diff)))
                rec["ort_vs_pt_aligned"] = True
            except Exception as eC:
                rec["ort_vs_pt_aligned"] = False
                rec["align_error"] = _tail(str(eC))
                rec["ort_out_shape"] = list(o.shape)
                rec["ort_out_dtype"] = str(o.dtype)
    except Exception as e:
        rec["ort_load_success"] = False
        rec["error_type"] = type(e).__name__
        rec["error_tail"] = _tail(str(e))
    return rec


for tag in ["legacy", "dynamo"]:
    if results.get(tag, {}).get("export_success"):
        results[tag + "_ort"] = _try_ort(tag)

with open("onnx_smoke_result.json", "w") as fp:
    json.dump(results, fp, indent=2)

# 摘要打印
print("=" * 60)
print("ONNX SMOKE 摘要")
print("=" * 60)
for tag in ["legacy", "dynamo"]:
    r = results.get(tag, {})
    print(f"[{tag}] export_success = {r.get('export_success')}")
    if not r.get("export_success"):
        print(f"   error: {r.get('error_type')}: {r.get('error_tail','')[:300]}")
    else:
        ort = results.get(tag + "_ort", {})
        print(f"   ort_load_success = {ort.get('ort_load_success')}")
        if ort.get("ort_load_success"):
            print(f"   inputs = {ort.get('inputs')}")
            print(f"   ort_vs_pt_max_abs_diff = {ort.get('ort_vs_pt_max_abs_diff','(未对齐): '+str(ort.get('align_error',''))[:200])}")
print("\n写入 onnx_smoke_result.json")
