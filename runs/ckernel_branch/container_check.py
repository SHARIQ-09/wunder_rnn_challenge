"""Inside the scorer-sim container: parity of each packaged solution against its Torch reference, then speed on real
weights, interleaved so a busy CPU affects all alike.
Usage: python runs/ckernel_branch/container_check.py NAME=DIR=REF.npz ... (DIR holds solution.py, wnn_step.c, weights.npz; REF may be empty)"""
import importlib.util, sys, time
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class DataPoint:
    seq_ix: int; step_in_seq: int; need_prediction: bool; state: np.ndarray


def load(d, name):
    sp = importlib.util.spec_from_file_location(name, f"{d}/solution.py")
    mod = importlib.util.module_from_spec(sp); sp.loader.exec_module(mod)
    return mod.PredictionModel()


models, refs = {}, {}
for i, item in enumerate(sys.argv[1:]):
    name, d, ref = item.split("=")
    models[name] = load(d, f"sol{i}")
    refs[name] = dict(np.load(ref)) if ref else None

for name, m in models.items():
    r = refs[name]
    if r is None:
        continue
    for j in range(3):
        raw, seq = r[f"raw{j}"], int(r[f"seq{j}"])
        mx = 0.0
        for s in range(len(raw)):
            p = m.predict(DataPoint(seq, s, s >= 99, raw[s]))
            if p is not None:
                mx = max(mx, float(np.abs(p - r[f"pred{j}"][s]).max()))
        print(f"{name} seq {j}: max |C - torch| = {mx:.2e}", flush=True)

raw = next(r for r in refs.values() if r is not None)["raw0"][:20000]
times = {k: [] for k in models}
for rnd in range(4):
    for k, m in models.items():
        t = time.perf_counter()
        for s in range(len(raw)):
            m.predict(DataPoint(1000 + rnd, s, s >= 99, raw[s]))
        times[k].append((time.perf_counter() - t) / len(raw) * 1e6)
for k, v in times.items():
    print(f"{k:10s} median {np.median(v):6.1f} us/step  (runs: {' '.join(f'{x:.1f}' for x in v)})", flush=True)
