"""Are column positions 0, 2, 4, 10 special? Scores every single column and every bid i - ask j volume pair against the
target and against the A+Br residual, then tests a held-out ridge correction. Column names follow src/features.py (first 11
= "bid", last 11 = "ask"); the organiser docs say an index is not a depth level.
Usage: python runs/levels/probe.py [n_valid_sequences]
"""
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from scipy.signal import lfilter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils import FEATURE_COLUMNS, GlobalAccumulator
from src.evaluate import load_pytorch_model
from src.feature_sets import build_features

torch.set_num_threads(2)
n_seq = int(sys.argv[1]) if len(sys.argv) > 1 else 200
TOP_N, RIDGE, ALPHAS = 20, 10.0, (0.05, 0.005)
models = [(load_pytorch_model(Path(c), torch.device("cpu")), fs) for c, fs in
          [("checkpoints_norm_ft/best_norm_ft.pt", "v144"), ("checkpoints_branch80_s21/best.pt", "raw_ev_ema")]]

NAMES = []
for inst in ("i0", "i1"):
    NAMES += [f"{inst}_p{k}" for k in range(22)] + [f"{inst}_v{k}" for k in range(22)]
    NAMES += [f"{inst}_b{i}-a{j}{s}" for s in ("", "_ema05", "_ema005") for i in range(11) for j in range(11)]
IDX = {n: k for k, n in enumerate(NAMES)}
PI, PJ = np.meshgrid(np.arange(11), np.arange(11), indexing="ij")
PI, PJ = PI.ravel(), PJ.ravel()


def ema(x, a):
    y, _ = lfilter([a], [1.0, a - 1.0], x, axis=0, zi=(1.0 - a) * x[:1])
    return y


def candidates(raw):
    out = []
    for pb, vb in ((0, 22), (52, 74)):
        pair = raw[:, vb + PI].astype(np.float64) - raw[:, vb + 11 + PJ].astype(np.float64)
        out += [raw[:, pb:pb + 22], raw[:, vb:vb + 22], pair, ema(pair, ALPHAS[0]), ema(pair, ALPHAS[1])]
    return np.hstack(out).astype(np.float32)


pf = pq.ParquetFile("datasets/valid.parquet")
rgs = np.linspace(0, pf.num_row_groups - 1, n_seq).astype(int)
total = 0
for rg in rgs:
    d = pf.read_row_group(int(rg), columns=["need_prediction", "is_scored"]).to_pandas()
    total += int((d.need_prediction.values & d.is_scored.values).sum())
F = np.empty((total, len(NAMES)), np.float32)
Y, P, G = np.empty((total, 2), np.float32), np.empty((total, 2), np.float32), np.empty(total, np.int32)
official, pos = GlobalAccumulator(), 0
with torch.no_grad():
    for k, rg in enumerate(rgs):
        d = pf.read_row_group(int(rg), columns=FEATURE_COLUMNS + ["t0", "t1", "need_prediction", "is_scored"]).to_pandas()
        raw = d[FEATURE_COLUMNS].to_numpy(np.float32)
        y = d[["t0", "t1"]].to_numpy(np.float32); m = d.need_prediction.values & d.is_scored.values
        pred = np.mean([mdl(torch.from_numpy(build_features(raw, fs)).unsqueeze(0))[0][0].float().numpy() for mdl, fs in models], 0)
        official.add(y, pred, m)
        n = int(m.sum())
        F[pos:pos + n] = candidates(raw)[m]; Y[pos:pos + n] = y[m]; P[pos:pos + n] = pred[m]; G[pos:pos + n] = k
        pos += n
assert pos == total and len(NAMES) == F.shape[1]


def wcorr(a, b, w):
    a = a - np.average(a, weights=w); b = b - np.average(b, weights=w)
    return float(np.sum(w * a * b) / np.sqrt(np.sum(w * a * a) * np.sum(w * b * b) + 1e-30))


def wp(y, p):
    return float(np.mean([wcorr(np.clip(p[:, j], -2, 2), np.clip(y[:, j], -2, 2), np.abs(np.clip(y[:, j], -2, 2))) for j in range(2)]))


print(f"{n_seq} valid sequences, {total:,} scored rows, {len(NAMES)} candidates", flush=True)
print(f"Metric check on this subset: own {wp(Y, P):.6f}, official {official.result()['weighted_pearson']:.6f}\n", flush=True)


def corr_all(sel, target, w):
    """|t|-weighted corr of every candidate with target, on the rows in sel."""
    w = w[sel].astype(np.float64); t = target[sel].astype(np.float64); W = w.sum()
    t = t - (w * t).sum() / W
    tt = (w * t * t).sum(); res = np.zeros(F.shape[1])
    for s in range(0, F.shape[1], 64):
        X = F[np.ix_(np.flatnonzero(sel), np.arange(s, min(s + 64, F.shape[1])))].astype(np.float64)
        X -= (w[:, None] * X).sum(0) / W
        res[s:s + X.shape[1]] = (w[:, None] * X * t[:, None]).sum(0) / np.sqrt((w[:, None] * X * X).sum(0) * tt + 1e-30)
    return res


half = {0: G % 2 == 0, 1: G % 2 == 1}
tc, rc = np.zeros((2, 2, len(NAMES))), np.zeros((2, 2, len(NAMES)))
for h, sel in half.items():
    for j in range(2):
        y = np.clip(Y[:, j], -2, 2).astype(np.float64); w = np.abs(y); p = P[:, j].astype(np.float64)
        beta = (w[sel] * y[sel] * p[sel]).sum() / (w[sel] * p[sel] ** 2).sum()
        tc[h, j] = corr_all(sel, y, w); rc[h, j] = corr_all(sel, y - beta * p, w)
    print(f"scored half {h}", flush=True)
mean_tc, mean_rc = tc.mean((0, 1)), rc.mean((0, 1))
stable = np.all(np.sign(rc) == np.sign(rc[0, 0]), axis=(0, 1))


def table(inst):
    names = [f"{inst}_b{i}-a{j}" for i, j in zip(PI, PJ)]
    a = np.abs(mean_tc[[IDX[n] for n in names]]); ar = np.abs(mean_rc[[IDX[n] for n in names]])
    tr, rr = (-a).argsort().argsort() + 1, (-ar).argsort().argsort() + 1
    print(f"\n{inst}: same-index pairs, rank among all 121 pairs (1 = strongest)")
    print("  pos   |corr target|  rank   |corr residual|  rank")
    for k in range(11):
        q = k * 11 + k
        print(f"  {k:3d}{' *' if k in (0, 2, 4, 10) else '  '}  {a[q]:11.4f}  {tr[q]:4d}   {ar[q]:13.4f}  {rr[q]:4d}")
    d = [k * 11 + k for k in (0, 2, 4, 10)]
    print(f"  median rank of positions 0,2,4,10: target {np.median(tr[d]):.0f}, residual {np.median(rr[d]):.0f}; median of all 121 is 61")
    print("  strongest pairs vs target:", ", ".join(f"{names[q]} {a[q]:.3f}" for q in (-a).argsort()[:5]))
    print("  mean |corr target| by bid position:", " ".join(f"{a[PI == k].mean():.3f}" for k in range(11)))
    print("  mean |corr target| by ask position:", " ".join(f"{a[PJ == k].mean():.3f}" for k in range(11)))


table("i0"); table("i1")
print("\nSingle volume columns, mean |corr target| by position 0-21")
for inst in ("i0", "i1"):
    print(f"  {inst}:", " ".join(f"{abs(mean_tc[IDX[f'{inst}_v{k}']]):.3f}" for k in range(22)))
print("\nStrongest candidates vs A+Br residual (same sign in both halves and both targets)")
for q in sorted(np.flatnonzero(stable), key=lambda q: -abs(mean_rc[q]))[:15]:
    print(f"  {NAMES[q]:24s} residual {mean_rc[q]:+.4f}  target {mean_tc[q]:+.4f}")

BASE = [IDX[f"{inst}_b{k}-a{k}{s}"] for inst in ("i0", "i1") for k in (0, 2, 4, 10) for s in ("", "_ema05", "_ema005")]
print("\nHeld-out ridge correction (columns chosen and fitted on one half, scored on the other)")
for fit in (0, 1):
    tr, te = half[fit], half[1 - fit]
    ti, te_i = np.flatnonzero(tr), np.flatnonzero(te)
    base = wp(Y[te], P[te])
    for label in ("positions 0,2,4,10", f"top {TOP_N} of all"):
        newp = P[te].copy()
        for j in range(2):
            cols = BASE if label.startswith("positions") else np.abs(rc[fit, j]).argsort()[-TOP_N:].tolist()
            y = np.clip(Y[tr, j], -2, 2).astype(np.float64); w = np.abs(y)
            A = F[np.ix_(ti, cols)].astype(np.float64); mu, sd = A.mean(0), A.std(0) + 1e-9
            X = np.column_stack([P[tr, j], (A - mu) / sd])
            coef = np.linalg.solve((X * w[:, None]).T @ X + RIDGE * np.eye(X.shape[1]), (X * w[:, None]).T @ y)
            B = (F[np.ix_(te_i, cols)].astype(np.float64) - mu) / sd
            newp[:, j] = np.column_stack([P[te, j], B]) @ coef
        new = wp(Y[te], newp)
        print(f"  fit half {fit}, {label:20s}: A+Br {base:.5f} -> {new:.5f}  ({new - base:+.5f})", flush=True)
