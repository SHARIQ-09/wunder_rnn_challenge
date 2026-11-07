"""Step-by-step reference for the 36 momentum features (spec for the C port); checks parity with batch.

Usage: python runs/momentum/parity_check.py [n_sequences]
"""
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils import FEATURE_COLUMNS
from src.feature_sets import build_features, _pressure_weights, N_EV_EMA, MOM_CLIP


class Online:
    def __init__(self):
        self.w = _pressure_weights()
        self.prev = self.prevP = self.P0 = None
        self.ef = self.es = None                       # EMAs of P started from the first value
        self.e2 = np.zeros(6); self.en = np.zeros(6); self.ed = 0.0
        self.w01 = self.w1 = 0.0                       # running weights for alpha 0.01 / 0.1
        self.anchors = deque(maxlen=5)
        self.signs = deque(); self.sign_sum = np.zeros(6)

    def __call__(self, x):
        w = self.w
        Vz = (x[22:44] - w["vmean"]) / w["vstd"]
        P4 = np.r_[w["wB"] @ Vz[:11], w["wS"] @ Vz[11:]]
        P = np.r_[P4, P4[:2] - P4[2:]]
        upd = self.prev is None or np.any(x[0:22] != self.prev[0:22])
        dP = np.zeros(6) if self.prevP is None else P - self.prevP
        if self.P0 is None:
            self.P0, self.ef, self.es = P.copy(), P.copy(), P.copy()
        else:
            self.ef = self.ef + 0.1 * (P - self.ef); self.es = self.es + 0.01 * (P - self.es)
        self.e2 = 0.99 * self.e2 + 0.01 * dP ** 2; self.w01 = 0.99 * self.w01 + 0.01
        scale = np.sqrt(self.e2 / self.w01) + 1e-3
        ts = x[48:52].max() - x[48:52].min()
        self.en = 0.9 * self.en + 0.1 * dP * ts; self.ed = 0.9 * self.ed + 0.1 * ts; self.w1 = 0.9 * self.w1 + 0.1
        trade_w = (self.en / self.w1) / (self.ed / self.w1 + 1e-3) / scale
        if upd:
            self.anchors.append(P.copy())
        m1 = (P - self.anchors[-1]) / scale
        m5 = (P - (self.anchors[0] if len(self.anchors) == 5 else self.P0)) / scale
        s = np.sign(dP); self.signs.append(s); self.sign_sum += s
        if len(self.signs) > 20:
            self.sign_sum -= self.signs.popleft()
        self.prev, self.prevP = x, P
        out = np.r_[(P - self.ef) / scale, (P - self.es) / scale, trade_w, m1, m5].clip(-MOM_CLIP, MOM_CLIP)
        return np.r_[out, self.sign_sum / 20.0]


n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
pf = pq.ParquetFile("datasets/valid.parquet")
worst = 0.0
for rg in range(n):
    raw = pf.read_row_group(rg, columns=FEATURE_COLUMNS).to_pandas().to_numpy(np.float32)
    batch = build_features(raw, "momentum")[:, 112 + N_EV_EMA:]
    o = Online()
    online = np.stack([o(r.astype(np.float64)) for r in raw])
    err = np.abs(batch - online).max(axis=0)
    worst = max(worst, err.max())
    print(f"seq {rg}: max err {err.max():.2e} (feature {err.argmax()}) | feature |max| {np.abs(batch).max():.1f} "
          f"| clipped share {(np.abs(batch[:, :30]) >= MOM_CLIP).mean():.4f}")
print(f"worst {worst:.2e}")
