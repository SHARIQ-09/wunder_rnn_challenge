"""Inside the container: time the C-kernel solution for ensembles of given sizes, using random weights.

Usage: python runs/ckernel/speed_probe.py "128:0,128:1" "96:0,96:1,96:1" ...
Each spec is a comma list of H:F (F 0 = v144 / 144 inputs, 1 = raw_ev_ema / 456 inputs), 2 layers each.
Times 20,000 real rows (valid sequence from runs/ckernel/reference_ens2.npz).
"""
import importlib.util, shutil, sys, tempfile, time
from dataclasses import dataclass
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
NF = {0: 144, 1: 456}


@dataclass(frozen=True)
class DataPoint:
    seq_ix: int; step_in_seq: int; need_prediction: bool; state: np.ndarray


def random_weights(spec, rng):
    w = {}
    for k, item in enumerate(spec.split(",")):
        H, F = map(int, item.split(":")); nf = NF[F]; p = f"m{k}_"
        r = lambda *s: (rng.standard_normal(s) * 0.05).astype(np.float32)
        w.update({p + "feat_mean": np.zeros(nf, np.float32), p + "feat_std": np.ones(nf, np.float32),
                  p + "input_proj.0.weight": r(H, nf), p + "input_proj.0.bias": r(H),
                  p + "input_proj.1.weight": np.ones(H, np.float32), p + "input_proj.1.bias": r(H)})
        for l in range(2):
            q = f"{p}gru_layers.{l}."
            w.update({q + "weight_ih_l0": r(3 * H, H), q + "weight_hh_l0": r(3 * H, H), q + "bias_ih_l0": r(3 * H), q + "bias_hh_l0": r(3 * H),
                      f"{p}norms.{l}.weight": np.ones(H, np.float32), f"{p}norms.{l}.bias": r(H)})
        w.update({p + "head.0.weight": np.ones(H, np.float32), p + "head.0.bias": r(H), p + "head.1.weight": r(H, H), p + "head.1.bias": r(H),
                  p + "head.4.weight": r(2, H), p + "head.4.bias": r(2), p + "H": np.array(H), p + "L": np.array(2), p + "F": np.array(F)})
    return w


raw = dict(np.load(HERE / "reference_ens2.npz"))["raw0"]
for spec in sys.argv[1:]:
    d = Path(tempfile.mkdtemp())
    for f in ("solution.py", "wnn_step.c"):
        shutil.copy(HERE / f, d / f)
    np.savez(d / "weights.npz", **random_weights(spec, np.random.default_rng(0)))
    sp = importlib.util.spec_from_file_location(f"sol_{abs(hash(spec))}", d / "solution.py")
    sol = importlib.util.module_from_spec(sp); sp.loader.exec_module(sol)
    m = sol.PredictionModel(d / "weights.npz")
    best = 1e9
    for rep in range(2):
        t = time.perf_counter()
        for s in range(len(raw)):
            m.predict(DataPoint(rep, s, s >= 99, raw[s]))
        best = min(best, (time.perf_counter() - t) / len(raw) * 1e6)
    print(f"{spec:28s} {best:6.1f} us/step", flush=True)
