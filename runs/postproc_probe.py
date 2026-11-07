"""Read-only: does using the event structure (targets change only on i0 price updates) help as post-processing?
Scores the ensemble's raw predictions vs. variants that pool predictions within each between-update segment."""
import sys, numpy as np, torch, pyarrow.parquet as pq
from pathlib import Path
from scipy.signal import lfilter
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import GlobalAccumulator, FEATURE_COLUMNS, TARGET_COLUMNS
from src.feature_sets import build_features
from src.evaluate import load_pytorch_model

dev = torch.device("cuda")
ckpts = ["checkpoints_norm_ft/best_norm_ft.pt", "checkpoints_raw_ev_ema_s21/best.pt"]
ms = [load_pytorch_model(Path(c), dev) for c in ckpts]
fsets = [torch.load(c, map_location="cpu").get("model_config", {}).get("feature_set", "v144") for c in ckpts]

def seg_ids(raw):
    u = np.ones(len(raw), bool); u[1:] = np.any(raw[1:, 0:22] != raw[:-1, 0:22], axis=1)
    return np.cumsum(u) - 1, u

def seg_cummean(P, seg):
    out = np.empty_like(P); s = np.r_[0, np.flatnonzero(np.diff(seg)) + 1, len(P)]
    for a, b in zip(s[:-1], s[1:]):
        out[a:b] = np.cumsum(P[a:b], 0) / np.arange(1, b - a + 1)[:, None]
    return out

def ema(P, a):
    return lfilter([a], [1, -(1 - a)], P, axis=0, zi=(1 - a) * P[:1])[0]

variants = ["raw", "hold_at_update", "cummean_in_segment", "ema0.5", "ema0.2", "0.5raw+0.5cummean"]
accs = {v: GlobalAccumulator() for v in variants}
pf = pq.ParquetFile("datasets/valid.parquet")
cols = FEATURE_COLUMNS + list(TARGET_COLUMNS) + ["need_prediction", "is_scored"]
with torch.no_grad():
    for rg in range(pf.num_row_groups):
        df = pf.read_row_group(rg, columns=cols).to_pandas()
        raw = df[FEATURE_COLUMNS].to_numpy(np.float32)
        X = {f: torch.from_numpy(build_features(raw, f)).unsqueeze(0).to(dev) for f in set(fsets)}
        P = np.mean([m(X[f])[0][0].float().cpu().numpy() for m, f in zip(ms, fsets)], axis=0).astype(np.float64)
        seg, u = seg_ids(raw)
        first = np.r_[0, np.flatnonzero(np.diff(seg)) + 1]
        cm = seg_cummean(P, seg)
        V = {"raw": P, "hold_at_update": P[first[seg]], "cummean_in_segment": cm,
             "ema0.5": ema(P, 0.5), "ema0.2": ema(P, 0.2), "0.5raw+0.5cummean": 0.5 * P + 0.5 * cm}
        y = df[list(TARGET_COLUMNS)].to_numpy(np.float32); mk = df.need_prediction.values & df.is_scored.values
        for v in variants:
            accs[v].add(y, V[v].astype(np.float32), mk)
for v in variants:
    r = accs[v].result(); print(f"{v:22s} WP={r['weighted_pearson']:.6f} t0={r['t0']:.6f} t1={r['t1']:.6f}")
