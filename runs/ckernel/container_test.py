"""Run inside a python:3.11-slim-bookworm container (gcc + numpy + onnxruntime installed).
1) Parity: C kernel vs PyTorch reference predictions on 3 validation sequences.
2) Latency: full predict() per step for the C kernel (1, 2, 3-model ensembles and H=192/256 shapes)
   versus v5's ONNX solution, all on one CPU, interleaved."""
import sys, time, importlib.util, numpy as np
from pathlib import Path
from dataclasses import dataclass

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


@dataclass(frozen=True)
class DataPoint:
    seq_ix: int
    step_in_seq: int
    need_prediction: bool
    state: np.ndarray


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


def run_seq(model, raw, seq, n=None):
    n = n or len(raw); P = np.zeros((n, 2), np.float32)
    for t in range(n):
        p = model.predict(DataPoint(seq, t, t >= 99, raw[t]))
        if p is not None: P[t] = p
    return P


def wp(y, p):
    y = np.clip(y, -2, 2).astype(np.float64); p = np.clip(p, -2, 2).astype(np.float64)
    w = np.abs(y); W = w.sum(); my = (w * y).sum() / W; mp = (w * p).sum() / W
    return float((w * (y - my) * (p - mp)).sum() / np.sqrt((w * (y - my) ** 2).sum() * (w * (p - mp) ** 2).sum()))


ref = np.load(HERE / "reference_normft.npz")
csol = load(HERE / "solution.py", "csol")

# 1) parity
m = csol.PredictionModel(HERE / "weights_normft.npz")
ys, pc, pt = [], [], []
for j in range(3):
    P = run_seq(m, ref[f"raw{j}"], int(ref[f"seq{j}"]))
    d = np.abs(P[99:] - ref[f"pred{j}"][99:]).max()
    print(f"seq {j}: max |C - torch| = {d:.2e}")
    mk = ref[f"mask{j}"]; ys.append(ref[f"y{j}"][mk]); pc.append(P[mk]); pt.append(ref[f"pred{j}"][mk])
Y, PC, PT = map(np.vstack, (ys, pc, pt))
print("WP C kernel :", np.mean([wp(Y[:, k], PC[:, k]) for k in range(2)]))
print("WP torch ref:", np.mean([wp(Y[:, k], PT[:, k]) for k in range(2)]))

# 2) latency: build synthetic weight files for other shapes from the real one
w = dict(np.load(HERE / "weights_normft.npz"))


def synth(H, L, K, path):
    rng = np.random.default_rng(0); out = {}
    for k in range(K):
        for name in csol.model_param_order("", L):
            src = w.get(f"m0_{name}")
            if name in ("feat_mean", "feat_std"):
                a = src
            elif name.endswith("weight") and ("norm" in name or name in ("input_proj.1.weight", "head.0.weight")):
                a = np.ones(H, np.float32)
            elif name.endswith("bias") and ("norm" in name or name in ("input_proj.1.bias", "head.0.bias")):
                a = np.zeros(H, np.float32)
            else:
                shape = {"input_proj.0.weight": (H, 144), "input_proj.0.bias": (H,), "head.1.weight": (H, H),
                         "head.1.bias": (H,), "head.4.weight": (2, H), "head.4.bias": (2,)}.get(name)
                if shape is None:
                    shape = (3 * H, H) if "weight" in name else (3 * H,)
                a = (rng.standard_normal(shape) * 0.05).astype(np.float32)
            out[f"m{k}_{name}"] = a
        out[f"m{k}_H"] = np.array(H); out[f"m{k}_L"] = np.array(L)
    np.savez(path, **out)


cases = {"C H128x2 (1 model)": (128, 2, 1), "C H128x2 (2-model ens)": (128, 2, 2), "C H128x2 (3-model ens)": (128, 2, 3),
         "C H192x2": (192, 2, 1), "C H256x2": (256, 2, 1), "C H128x3": (128, 3, 1)}
models = {}
for name, (H, L, K) in cases.items():
    p = Path("/tmp") / f"w_{H}_{L}_{K}.npz"; synth(H, L, K, p); models[name] = csol.PredictionModel(p)
v5 = load(ROOT / "submission_5epoch" / "solution.py", "v5sol")
models["v5 ONNX (reference)"] = v5.PredictionModel()
raw = ref["raw0"]; N = 8000
times = {k: [] for k in models}
for rnd in range(4):
    for k, mdl in models.items():
        mdl.current_seq_ix = None
        t0 = time.perf_counter_ns(); run_seq(mdl, raw, 10_000 + rnd, N)
        times[k].append((time.perf_counter_ns() - t0) / N / 1000)
base = np.median(times["v5 ONNX (reference)"])
for k, v in times.items():
    print(f"{k:26s} {np.median(v):6.1f} us/step  ratio vs v5 {np.median(v) / base:4.2f}")
