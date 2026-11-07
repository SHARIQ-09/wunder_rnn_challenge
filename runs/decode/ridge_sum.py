"""Loose end from Step 3 vs Step 4: ridge on the price row had sum corr 0.59 in unit 0.6569 (all rows), LightGBM 0.10 (scored rows).
Same ridge scored on the rows of exploit_preds.npz, all and scored, plus held-out sum swap and per-population oracles."""
import sys
from pathlib import Path
import numpy as np, pyarrow.parquet as pq
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils import FEATURE_COLUMNS as F
UNITS = (0.6569, 0.2413, 0.46855, 0.12543); NV = 300
Z = np.load("runs/decode/exploit_preds.npz"); P, Y, L, G = Z["P"], Z["Y"], Z["L"], Z["G"]
lab = {s: np.load(f"runs/decode/{s}_unit.npy") for s in ("train", "valid")}
pf = {s: pq.ParquetFile(f"datasets/{s}.parquet") for s in ("train", "valid")}


def wc(p, y):
    p = np.clip(p, -2, 2); y = np.clip(y, -2, 2); w = np.abs(y)
    a = p - np.average(p, weights=w); b = y - np.average(y, weights=w)
    return float((w * a * b).sum() / np.sqrt((w * a * a).sum() * (w * b * b).sum() + 1e-30))
wp = lambda P, Y: float(np.mean([wc(P[:, j], Y[:, j]) for j in range(2)]))
def rowx(d):
    return d - d.mean(1, keepdims=True)

coef = {}
for k in range(4):
    X, S = [], []
    for rg in np.random.default_rng(k).choice(np.flatnonzero(lab["train"] == k), 60, replace=False):
        d = pf["train"].read_row_group(int(rg), columns=F[:22] + ["t0", "t1", "need_prediction"]).to_pandas(); n = d.need_prediction.values
        X.append(rowx(d[F[:22]].to_numpy(np.float64))[n][::4]); S.append((d.t0.values + d.t1.values)[n][::4])
    X, S = np.vstack(X), np.concatenate(S); mx, my = X.mean(0), S.mean(); A = X - mx
    coef[k] = (mx, my, np.linalg.solve(A.T @ A + 1e-3 * len(A) * A.var(0).mean() * np.eye(22), A.T @ (S - my)))

R, seq_all = [], {k: [] for k in range(4)}
rgs = np.linspace(0, pf["valid"].num_row_groups - 1, NV).astype(int)
for i in np.unique(G):
    rg = int(rgs[i]); k = int(lab["valid"][rg]); d = pf["valid"].read_row_group(rg, columns=F[:22] + ["t0", "t1", "need_prediction", "is_scored"]).to_pandas()
    n = d.need_prediction.values; mk = n & d.is_scored.values; mx, my, b = coef[k]; r = (rowx(d[F[:22]].to_numpy(np.float64)) - mx) @ b + my
    R.append(r[mk]); s = (d.t0.values + d.t1.values)[n][::20]; seq_all[k].append((r[n][::20], s, np.full(len(s), i)))
R = np.concatenate(R)
for k, u in enumerate(UNITS):
    s = L == k; a, b, q = (np.concatenate([x[j] for x in seq_all[k]]) for j in range(3))
    within = np.mean([np.corrcoef(a[q == i], b[q == i])[0, 1] for i in np.unique(q) if b[q == i].std() > 0 and a[q == i].std() > 0])
    d = P[s, 0] - P[s, 1]; T = Y[s].sum(1); ns = R[s]
    swap = np.column_stack([(d + ns) / 2, (ns - d) / 2]); orac = np.column_stack([(d + T) / 2, (T - d) / 2])
    print(f"unit {u:.4f}: ridge sum corr all rows {np.corrcoef(a, b)[0, 1]:.4f} scored {np.corrcoef(R[s], T)[0, 1]:.4f} within-seq all rows {within:.4f} | A+B scored {np.corrcoef(P[s].sum(1), T)[0, 1]:.4f}"
          f" | WP base {wp(P[s], Y[s]):.4f} ridge sum {wp(swap, Y[s]):.4f} oracle sum {wp(orac, Y[s]):.4f}")
for par in (0, 1):
    a_ = G % 2 != par; Qp = P[a_].copy()
    for k in (0,):
        s = L[a_] == k; d = Qp[s, 0] - Qp[s, 1]; Qp[s] = np.column_stack([(d + R[a_][s]) / 2, (R[a_][s] - d) / 2])
    print(f"global, half {par}, ridge sum only in unit 0.6569: {wp(P[a_], Y[a_]):.5f} -> {wp(Qp, Y[a_]):.5f}")
