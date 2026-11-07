"""Step-by-step reference for the 3 kept + 49 pressure features (spec for the C port); checks parity with batch.

Usage: python runs/pressure/parity_check.py [n_sequences]
"""
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils import FEATURE_COLUMNS
from src.feature_sets import build_features, _pressure_weights, N_EV_EMA


class Ema:
    def __init__(self, a):
        self.a, self.v = a, None

    def __call__(self, x):
        self.v = x if self.v is None else self.v + self.a * (x - self.v)
        return self.v


class Window:
    def __init__(self, n):
        self.q, self.n, self.s = deque(), n, 0.0

    def __call__(self, x):
        self.q.append(x); self.s += x
        if len(self.q) > self.n:
            self.s -= self.q.popleft()
        return self.s


class Online:
    def __init__(self):
        self.w = _pressure_weights()
        self.step, self.prev = 0, None
        self.e_dv, self.e_mf = Ema(.05), Ema(.05)
        self.e_pf, self.e_ps = Ema(.1), Ema(.01)
        self.t10, self.t50, self.u10, self.u50, self.big = Window(10), Window(50), Window(10), Window(50), Window(1000)
        self.mv_s, self.mv_f, self.u_s, self.u_f, self.tr_s, self.tr_f = (Ema(a) for a in (.01, .1, .01, .1, .01, .1))
        self.e_fam5, self.e_fam1, self.e_a1 = Ema(.05), Ema(.01), Ema(.1)
        self.scale2 = float(self.w["mv2_init"])
        self.last_trade = self.last_big = -1
        self.last_upd, self.P4_upd, self.last_move = 0, None, 0.0
        self.moves = deque(maxlen=5)
        self.prev_a6 = None

    def __call__(self, x):
        w, t = self.w, self.step
        upd = self.prev is None or np.any(x[0:22] != self.prev[0:22])
        trade = bool(np.any(x[48:52] != x[48]))
        dv, mf = x[48:52].sum(), (x[44:48] * x[48:52]).sum()
        keep = [mf, self.e_dv(dv), self.e_mf(mf)]

        Vz = (x[22:44] - w["vmean"]) / w["vstd"]
        P4 = np.r_[w["wB"] @ Vz[:11], w["wS"] @ Vz[11:]]
        pstep = 0.0 if self.prev is None else float(((x[0:22] - self.prev[0:22]) / w["dstd"]) @ w["pcomp"])
        if upd:
            self.last_upd, self.P4_upd, self.last_move = t, P4.copy(), pstep
            self.moves.append(pstep)
        since_upd = t - self.last_upd
        net = P4[:2] - P4[2:]
        share = P4[:2] / (np.abs(P4[:2]) + np.abs(P4[2:]) + 0.1)
        d_since = P4 - self.P4_upd
        g1 = np.r_[P4, net, share, d_since, self.e_pf(P4), self.e_ps(P4), d_since * np.log1p(since_upd)]

        if trade:
            self.last_trade = t
        tf, uf = float(trade), float(upd)
        t10, t50, u10, u50 = self.t10(tf), self.t50(tf), self.u10(uf), self.u50(uf)
        g2 = [t10, t50, np.log1p(t - self.last_trade), u10, u50,
              np.log((t10 / 10 + .01) / (t50 / 50 + .01)), np.log((u10 / 10 + .01) / (u50 / 50 + .01))]

        mv = abs(pstep)
        shock = min(mv / (np.sqrt(self.scale2) + 1e-6), 50.0)
        self.scale2 += 0.005 * (mv * mv - self.scale2)
        big = shock > 6.0
        if big:
            self.last_big = t
        mv_s, u_s, tr_s = self.mv_s(mv), self.u_s(uf), self.tr_s(tf)
        g3 = [mv_s, u_s, tr_s, np.log1p(mv_s / (tr_s + .01)),
              np.log((self.mv_f(mv) + 1e-3) / (mv_s + 1e-3)), np.log((self.u_f(uf) + 1e-3) / (u_s + 1e-3)),
              np.log((self.tr_f(tf) + 1e-3) / (tr_s + 1e-3)), shock, np.log1p(t - self.last_big), self.big(float(big))]

        a = x[104:112]
        fam = (a[0] + a[4] - a[7]) / 3.0
        da6 = 0.0 if self.prev_a6 is None else a[6] - self.prev_a6
        self.prev_a6 = a[6]
        g4 = np.r_[self.last_move, sum(self.moves), self.last_move * net, self.e_fam5(fam), self.e_fam1(fam), self.e_a1(a[1]), da6]

        self.prev, self.step = x.copy(), t + 1
        return np.r_[keep, g1, g2, g3, g4]


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    pf = pq.ParquetFile("datasets/valid.parquet")
    worst = 0.0
    for i in range(n):
        raw = pf.read_row_group(i * 500, columns=FEATURE_COLUMNS).to_pandas().to_numpy(np.float32)
        batch = build_features(raw, "pressure")[:, 112 + N_EV_EMA:].astype(np.float64)
        on = Online()
        online = np.stack([on(r.astype(np.float64)) for r in raw])
        err = np.abs(batch - online) / (1.0 + np.abs(online))
        col = int(err.max(0).argmax())
        print(f"seq rg {i*500}: max rel err {err.max():.2e} (feature {col})")
        worst = max(worst, err.max())
    print("PARITY OK" if worst < 1e-4 else "PARITY FAILED")
