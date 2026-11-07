"""Is t0+t1 a function of the current i0 price row, and how much would knowing it be worth?
(a) determinism: within a sequence, rows with an identical i0 price row, first half vs second half
(b) A+B's skill on the sum
(c) oracle: keep the model's t0-t1, replace its t0+t1 with the truth (upper bound on the gain)
"""
import sys
from pathlib import Path
import numpy as np, pyarrow.parquet as pq, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils import FEATURE_COLUMNS as F
from src.evaluate import load_pytorch_model
from src.feature_sets import build_features
dev = torch.device("cuda")
ms = [(load_pytorch_model(Path("checkpoints_norm_ft/best_norm_ft.pt"), dev), "v144"), (load_pytorch_model(Path("checkpoints_raw_ev_ema_s21/best.pt"), dev), "raw_ev_ema")]
def wc(p, y):
    p = np.clip(p, -2, 2); y = np.clip(y, -2, 2); w = np.abs(y)
    a = p - np.average(p, weights=w); b = y - np.average(y, weights=w)
    return float((w * a * b).sum() / np.sqrt((w * a * a).sum() * (w * b * b).sum() + 1e-30))
wp = lambda P, Y: np.mean([wc(P[:, j], Y[:, j]) for j in range(2)])
pf = pq.ParquetFile("datasets/valid.parquet"); G = np.load("runs/groups/valid_group.npy")
P, Y, Gs, det = [], [], [], []
with torch.no_grad():
    for rg in np.linspace(0, pf.num_row_groups - 1, 150).astype(int):
        d = pf.read_row_group(int(rg), columns=F + ["t0", "t1", "need_prediction", "is_scored"]).to_pandas()
        raw = d[F].to_numpy(np.float32); y = d[["t0", "t1"]].to_numpy(np.float64); s = y.sum(1)
        p = np.mean([m(torch.from_numpy(build_features(raw, fs)).unsqueeze(0).to(dev))[0][0].float().cpu().numpy() for m, fs in ms], 0)
        mk = d.need_prediction.values & d.is_scored.values
        P.append(p[mk]); Y.append(y[mk]); Gs.append(np.full(mk.sum(), G[rg]))
        n = d.need_prediction.values; idx = np.flatnonzero(n); key = [raw[i, :22].tobytes() for i in idx]
        seen = {}; agree = tot = 0
        for i, k in zip(idx, key):
            if k in seen:
                tot += 1; agree += abs(seen[k] - s[i]) < 1e-3
            else:
                seen[k] = s[i]
        det.append((agree, tot, len(seen)))
P, Y, Gs = map(np.concatenate, (P, Y, Gs))
a, t, u = np.sum(det, 0)
print(f"(a) repeat price rows: {t:,} over {u:,} distinct; same t0+t1 as first time seen: {a / max(t,1):.4f}")
for nm, sel in [("all", np.ones(len(Y), bool))] + [(f"group {k}", Gs == k) for k in range(3)]:
    s = Y[sel].sum(1); ps = P[sel].sum(1); d = P[sel, 0] - P[sel, 1]
    oracle = np.column_stack([(d + s) / 2, (s - d) / 2])
    dtrue = Y[sel, 0] - Y[sel, 1]; odiff = np.column_stack([(dtrue + ps) / 2, (ps - dtrue) / 2])
    print(f"(b,c) {nm}: corr(pred sum, true sum) {np.corrcoef(ps, s)[0,1]:.4f}  corr(pred diff, true diff) {np.corrcoef(d, dtrue)[0,1]:.4f} | WP base {wp(P[sel], Y[sel]):.4f}  true sum {wp(oracle, Y[sel]):.4f}  true diff {wp(odiff, Y[sel]):.4f}", flush=True)
