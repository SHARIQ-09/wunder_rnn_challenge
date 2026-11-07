"""XGBoost on top of a GRU (stacking): does a tree model on the GRU's memory beat the GRU alone?

Inputs per scored row: GRU prediction (2), GRU last-layer memory (128), raw row (112).
Fit on half of a validation sample, score on the other half (split by sequence), official mask.
"""
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils import FEATURE_COLUMNS
from src.evaluate import load_pytorch_model
from src.feature_sets import build_features

CKPT = sys.argv[1] if len(sys.argv) > 1 else "checkpoints_raw_ev_ema_s21/best.pt"
N_SEQ, STEP = 200, 3
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_pytorch_model(Path(CKPT), dev)
fset = torch.load(CKPT, map_location="cpu").get("model_config", {}).get("feature_set", "v144")


@torch.no_grad()
def run_gru(x):
    """Same as ScaledWunderGRU.forward, but also returns the last layer's memory at every step."""
    m = model
    x = (x - m.feat_mean) / m.feat_std
    if float(m.norm_clip) > 0:
        x = torch.clamp(x, -float(m.norm_clip), float(m.norm_clip))
    h = m.input_proj(x)
    for i, gru in enumerate(m.gru_layers):
        out, _ = gru(h)
        h = m.norms[i](out) if i == 0 else m.norms[i](out + h)
    pred, _ = m(x * m.feat_std + m.feat_mean)
    return pred[0].float().cpu().numpy(), h[0].float().cpu().numpy()


pf = pq.ParquetFile("datasets/valid.parquet")
ids = np.sort(np.random.default_rng(11).choice(pf.num_row_groups, N_SEQ, replace=False))
P, H, R, Y, G = [], [], [], [], []
for g, i in enumerate(ids):
    d = pf.read_row_group(int(i), columns=FEATURE_COLUMNS + ["t0", "t1", "need_prediction", "is_scored"]).to_pandas()
    raw = d[FEATURE_COLUMNS].to_numpy(np.float32)
    p, h = run_gru(torch.from_numpy(build_features(raw, fset)).unsqueeze(0).to(dev))
    rows = np.flatnonzero(d.need_prediction.values & d.is_scored.values)[::STEP]
    P.append(p[rows]); H.append(h[rows].astype(np.float16)); R.append(raw[rows])
    Y.append(np.clip(d[["t0", "t1"]].to_numpy(np.float32)[rows], -2, 2)); G.append(np.full(len(rows), g))
P, H, R, Y, G = np.vstack(P), np.vstack(H).astype(np.float32), np.vstack(R), np.vstack(Y), np.concatenate(G)
fit, test = G < N_SEQ // 2, G >= N_SEQ // 2
inner = fit & (G >= int(N_SEQ * 0.4))          # last 20% of fit sequences: early stopping only
print(f"[{fset}] {CKPT}: rows fit {fit.sum()} test {test.sum()}", flush=True)


def wp(y, p):
    w = np.abs(y); my, mp = np.average(y, weights=w), np.average(p, weights=w)
    return np.sum(w * (y - my) * (p - mp)) / np.sqrt(np.sum(w * (y - my) ** 2) * np.sum(w * (p - mp) ** 2))


def stack(X, j, from_gru):
    tr, ev = fit & ~inner, inner
    kw = dict(base_margin=P[:, j]) if from_gru else {}
    dtr = xgb.DMatrix(X[tr], Y[tr, j], weight=np.abs(Y[tr, j]) + 1e-3, **({"base_margin": P[tr, j]} if from_gru else {}))
    dev_ = xgb.DMatrix(X[ev], Y[ev, j], weight=np.abs(Y[ev, j]) + 1e-3, **({"base_margin": P[ev, j]} if from_gru else {}))
    dte = xgb.DMatrix(X[test], **({"base_margin": P[test, j]} if from_gru else {}))
    params = dict(objective="reg:squarederror", eta=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.5,
                  min_child_weight=50, tree_method="hist", nthread=4)
    b = xgb.train(params, dtr, 1000, evals=[(dev_, "ev")], early_stopping_rounds=50, verbose_eval=False)
    return wp(Y[test, j], b.predict(dte, iteration_range=(0, b.best_iteration + 1))), b.best_iteration + 1


base = [wp(Y[test, j], P[test, j]) for j in range(2)]
print(f"{'GRU alone':52s} t0 {base[0]:.4f}  t1 {base[1]:.4f}  WP {np.mean(base):.4f}", flush=True)
for name, X, from_gru in [
    ("XGB on GRU prediction only", P, False),
    ("XGB on GRU memory (128) + prediction", np.hstack([H, P]), False),
    ("XGB on memory + prediction + raw row", np.hstack([H, P, R]), False),
    ("XGB correcting GRU (starts from GRU pred): memory+raw", np.hstack([H, R]), True),
]:
    s = [stack(X, j, from_gru) for j in range(2)]
    print(f"{name:52s} t0 {s[0][0]:.4f}  t1 {s[1][0]:.4f}  WP {np.mean([s[0][0], s[1][0]]):.4f}"
          f"  (trees {s[0][1]}/{s[1][1]}, change {np.mean([s[0][0], s[1][0]]) - np.mean(base):+.4f})", flush=True)
