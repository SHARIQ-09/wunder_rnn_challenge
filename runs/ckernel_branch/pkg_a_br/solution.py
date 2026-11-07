"""Competition entry point using a C kernel compiled at load time (gcc is in the scorer image).

Files next to this module: wnn_step.c, weights.npz (one or more models: m0_*, m1_*, ...).
Per step: one ctypes call computes the feature sets the models need (m{k}_F: 0 = v144,
1 = raw_ev_ema), every model in the ensemble, and the averaged prediction.
Models with m{k}_T = 1 are BranchGRU (always raw_ev_ema).
"""
from __future__ import annotations

import ctypes
import os
import subprocess
import tempfile
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

HERE = Path(__file__).resolve().parent


def _compile() -> ctypes.CDLL:
    out_dir = Path(tempfile.mkdtemp(prefix="wnn_"))
    so = out_dir / "wnn_step.so"
    cmd = ["gcc", "-O3", "-march=native", "-funroll-loops", "-shared", "-fPIC", str(HERE / "wnn_step.c"), "-o", str(so), "-lm"]
    subprocess.run(cmd, check=True, capture_output=True)
    return ctypes.CDLL(str(so))


def model_param_order(prefix: str, L: int):
    names = ["feat_mean", "feat_std", "input_proj.0.weight", "input_proj.0.bias",
             "input_proj.1.weight", "input_proj.1.bias"]
    for l in range(L):
        names += [f"gru_layers.{l}.weight_ih_l0", f"gru_layers.{l}.weight_hh_l0",
                  f"gru_layers.{l}.bias_ih_l0", f"gru_layers.{l}.bias_hh_l0",
                  f"norms.{l}.weight", f"norms.{l}.bias"]
    names += ["head.0.weight", "head.0.bias", "head.1.weight", "head.1.bias", "head.4.weight", "head.4.bias"]
    return [f"{prefix}{n}" for n in names]


def branch_param_order(prefix: str):
    gru = lambda g: [f"{g}.weight_ih_l0", f"{g}.weight_hh_l0", f"{g}.bias_ih_l0", f"{g}.bias_hh_l0"]
    names = ["feat_mean", "feat_std"]
    for b in range(4):
        names += [f"branch_proj.{b}.0.weight", f"branch_proj.{b}.0.bias", f"branch_proj.{b}.1.weight", f"branch_proj.{b}.1.bias"]
        names += gru(f"branch_gru.{b}") + [f"branch_norm.{b}.weight", f"branch_norm.{b}.bias"]
    names += ["fusion_proj.0.weight", "fusion_proj.0.bias", "fusion_proj.1.weight", "fusion_proj.1.bias"]
    names += gru("fusion_gru") + ["fusion_norm.weight", "fusion_norm.bias"]
    names += ["head.0.weight", "head.0.bias", "head.1.weight", "head.1.bias", "head.4.weight", "head.4.bias"]
    return [f"{prefix}{n}" for n in names]


class PredictionModel:
    def __init__(self, weights_path: str | os.PathLike | None = None):
        self.lib = _compile()
        f32p = ctypes.POINTER(ctypes.c_float)
        self.lib.wnn_add_model.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(f32p), f32p]
        self.lib.wnn_add_model.restype = ctypes.c_int
        self.lib.wnn_step.argtypes = [f32p, f32p]
        i32p = ctypes.POINTER(ctypes.c_int)
        self.lib.wnn_add_branch.argtypes = [ctypes.c_int, ctypes.POINTER(f32p), ctypes.POINTER(i32p), i32p, f32p]
        self.lib.wnn_add_branch.restype = ctypes.c_int
        w = dict(np.load(weights_path or (HERE / "weights.npz")))
        self._keep = []
        k = 0
        while f"m{k}_H" in w:
            H, L = int(w[f"m{k}_H"]), int(w[f"m{k}_L"])
            if int(w.get(f"m{k}_T", 0)) == 1:
                arrs = [np.ascontiguousarray(w[n].T if w[n].ndim == 2 else w[n], dtype=np.float32)
                        for n in branch_param_order(f"m{k}_")]
                idx = [np.ascontiguousarray(w[f"m{k}_idx{b}"], dtype=np.int32) for b in range(4)]
                n = np.array([len(a) for a in idx], dtype=np.int32)
                hstate = np.zeros(H * 5, dtype=np.float32)
                ptrs = (f32p * len(arrs))(*[a.ctypes.data_as(f32p) for a in arrs])
                iptrs = (i32p * 4)(*[a.ctypes.data_as(i32p) for a in idx])
                self._keep += arrs + idx + [n, hstate, ptrs, iptrs]
                if self.lib.wnn_add_branch(H, ptrs, iptrs, n.ctypes.data_as(i32p), hstate.ctypes.data_as(f32p)) < 0:
                    raise RuntimeError("branch model rejected by kernel")
                k += 1
                continue
            F = int(w.get(f"m{k}_F", 0))
            # 2-D weight matrices are passed transposed (cols x rows) for the vectorized C matvec.
            arrs = [np.ascontiguousarray(w[n].T if w[n].ndim == 2 else w[n], dtype=np.float32)
                    for n in model_param_order(f"m{k}_", L)]
            hstate = np.zeros(H * L, dtype=np.float32)
            ptrs = (f32p * len(arrs))(*[a.ctypes.data_as(f32p) for a in arrs])
            self._keep += arrs + [hstate, ptrs]
            if self.lib.wnn_add_model(H, L, F, ptrs, hstate.ctypes.data_as(f32p)) < 0:
                raise RuntimeError("model rejected by kernel")
            k += 1
        if k == 0:
            raise RuntimeError("no models in weights file")
        self.out = np.zeros(2, dtype=np.float32)
        self._out_p = self.out.ctypes.data_as(f32p)
        self._step = self.lib.wnn_step
        self._f32p = f32p
        self.current_seq_ix = None
        self.previous_step = None

    def predict(self, data_point):
        seq_ix = int(data_point.seq_ix)
        step = int(data_point.step_in_seq)
        if self.current_seq_ix != seq_ix:
            if step != 0:
                raise ValueError("a new sequence must start at step zero")
            self.current_seq_ix = seq_ix
            self.previous_step = None
            self.lib.wnn_reset()
        if self.previous_step is not None and step != self.previous_step + 1:
            raise ValueError("rows must arrive in sequence order")
        self.previous_step = step
        state = np.ascontiguousarray(data_point.state, dtype=np.float32)
        self._step(state.ctypes.data_as(self._f32p), self._out_p)
        if not data_point.need_prediction:
            return None
        return self.out.copy()
