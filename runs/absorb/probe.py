"""Price-move -> book-response (absorption) probe.

1. Does the relation exist? After an i0 price move, how do bid and ask volume change over the next steps?
2. Does it carry signal the trained A+B ensemble misses? |t|-weighted correlation of each feature with the
   ensemble residual, and held-out WP gain of a ridge correction (fit on half the sequences, scored on the other).
Usage: python runs/absorb/probe.py [n_valid_sequences]
"""
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils import FEATURE_COLUMNS, GlobalAccumulator
from src.evaluate import load_pytorch_model
from src.feature_sets import build_features, _pressure_weights, _ema

n_seq = int(sys.argv[1]) if len(sys.argv) > 1 else 160
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models = [(load_pytorch_model(Path(c), dev), fs) for c, fs in
          [("checkpoints_norm_ft/best_norm_ft.pt", "v144"), ("checkpoints_raw_ev_ema_s21/best.pt", "raw_ev_ema")]]
w = _pressure_weights()
NAMES = ["move", "dBid", "dAsk", "dB_score", "dS_score",
         "dBid*down", "dAsk*up", "dBid*up", "dAsk*down", "dBid*move", "dAsk*move",
         "absorb_net", "absorb_net*|move|", "dB_score*move", "dS_score*move", "ema(absorb_net)"]


def feats(raw):
    x = raw.astype(np.float64)
    L = len(x); idx = np.arange(L)
    upd = np.ones(L, bool); upd[1:] = np.any(x[1:, :22] != x[:-1, :22], axis=1)
    dp = np.zeros(L); dp[1:] = (np.diff(x[:, :22], axis=0) / w["dstd"]).mean(1)      # mean price-column move: + = up
    last = np.maximum.accumulate(np.where(upd, idx, 0))
    move = dp[last]                                                                    # size/sign of the last price update
    Vz = (x[:, 22:44] - w["vmean"]) / w["vstd"]
    bid, ask = Vz[:, :11].mean(1), Vz[:, 11:].mean(1)
    P4 = np.column_stack([Vz[:, :11] @ w["wB"].T, Vz[:, 11:] @ w["wS"].T])
    Bs, Ss = P4[:, 0], P4[:, 2]
    dBid, dAsk = bid - bid[last], ask - ask[last]                                      # book response since that move
    dBs, dSs = Bs - Bs[last], Ss - Ss[last]
    up, down = (move > 0).astype(float), (move < 0).astype(float)
    absorb = dBid * down - dAsk * up                                                   # + = bids absorbing a drop / asks NOT absorbing a rise
    F = np.column_stack([move, dBid, dAsk, dBs, dSs, dBid * down, dAsk * up, dBid * up, dAsk * down,
                         dBid * move, dAsk * move, absorb, absorb * np.abs(move), dBs * move, dSs * move, _ema(absorb, 0.05)])
    return F, upd, dp, bid, ask


pf = pq.ParquetFile("datasets/valid.parquet")
rel = {k: [] for k in ("mv", "db", "da")}
Y, P, F, M, G = [], [], [], [], []
with torch.no_grad():
    for k, rg in enumerate(np.linspace(0, pf.num_row_groups - 1, n_seq).astype(int)):
        d = pf.read_row_group(int(rg), columns=FEATURE_COLUMNS + ["t0", "t1", "need_prediction", "is_scored"]).to_pandas()
        raw = d[FEATURE_COLUMNS].to_numpy(np.float32)
        pred = np.mean([m(torch.from_numpy(build_features(raw, fs)).unsqueeze(0).to(dev))[0][0].float().cpu().numpy()
                        for m, fs in models], 0)
        f, upd, dp, bid, ask = feats(raw)
        pos = np.flatnonzero(upd[1:]) + 1
        pos = pos[pos < len(raw) - 20]
        rel["mv"].append(dp[pos]); rel["db"].append(bid[pos + 20] - bid[pos - 1]); rel["da"].append(ask[pos + 20] - ask[pos - 1])
        m = d.need_prediction.values & d.is_scored.values
        Y.append(d[["t0", "t1"]].to_numpy(np.float32)[m]); P.append(pred[m]); F.append(f[m]); G.append(np.full(m.sum(), k))
Y, P, F, G = map(np.concatenate, (Y, P, F, G))
mv, db, da = (np.concatenate(rel[k]) for k in ("mv", "db", "da"))

print(f"{n_seq} valid sequences, {len(Y):,} scored rows\n")
print("1. Relation: book change over the 20 steps after a price update")
for name, sel in [("price DOWN", mv < 0), ("price UP", mv > 0)]:
    print(f"   {name:10s} n={sel.sum():7d}  mean d(bid vol)={db[sel].mean():+.4f}  mean d(ask vol)={da[sel].mean():+.4f}")
print(f"   corr(move, d bid vol)={np.corrcoef(mv, db)[0,1]:+.3f}   corr(move, d ask vol)={np.corrcoef(mv, da)[0,1]:+.3f}\n")


def wcorr(a, b, wt):
    a = a - np.average(a, weights=wt); b = b - np.average(b, weights=wt)
    return np.sum(wt * a * b) / np.sqrt(np.sum(wt * a * a) * np.sum(wt * b * b) + 1e-30)


print("2. |t|-weighted corr with target and with A+B residual")
Fs = (F - F.mean(0)) / (F.std(0) + 1e-12)
for j in range(2):
    y = np.clip(Y[:, j], -2, 2); wt = np.abs(Y[:, j])
    beta = np.sum(wt * y * P[:, j]) / np.sum(wt * P[:, j] ** 2)
    r = y - beta * P[:, j]
    print(f"   t{j}: corr(pred, t)={wcorr(P[:, j], y, wt):.4f}")
    for i, nm in enumerate(NAMES):
        print(f"      {nm:18s} target {wcorr(Fs[:, i], y, wt):+.4f}   residual {wcorr(Fs[:, i], r, wt):+.4f}")

print("\n3. Held-out ridge correction on the residual (fit on even sequences, score odd ones)")


def wp(y, p):
    """Mean over t0/t1 of |t|-weighted Pearson with predictions clipped to +-2 (metric definition)."""
    return np.mean([wcorr(np.clip(p[:, j], -2, 2), y[:, j], np.abs(y[:, j])) for j in range(2)])


for subset, cols in [("all 16", list(range(len(NAMES)))), ("ema(absorb_net) only", [NAMES.index("ema(absorb_net)")])]:
    for train_par in (0, 1):
        tr, te = (G % 2 == train_par), (G % 2 != train_par)
        newp = P[te].copy()
        for j in range(2):
            y = np.clip(Y[tr, j], -2, 2); wt = np.abs(Y[tr, j])
            X = np.column_stack([P[tr, j], Fs[tr][:, cols]])
            coef = np.linalg.solve((X * wt[:, None]).T @ X + 10.0 * np.eye(X.shape[1]), (X * wt[:, None]).T @ y)
            newp[:, j] = np.column_stack([P[te, j], Fs[te][:, cols]]) @ coef
        print(f"   {subset:22s} fit on {'even' if train_par == 0 else 'odd '}: A+B {wp(Y[te], P[te]):.5f} -> corrected {wp(Y[te], newp):.5f}")
