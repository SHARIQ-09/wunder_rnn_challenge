"""Valid WP per group on scored rows vs all need_prediction rows, and scored-row counts."""
import sys
from pathlib import Path
import numpy as np, pyarrow.parquet as pq, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils import FEATURE_COLUMNS
from src.evaluate import load_pytorch_model
from src.feature_sets import build_features
dev = torch.device("cuda")
ms = [(load_pytorch_model(Path("checkpoints_norm_ft/best_norm_ft.pt"), dev), "v144"), (load_pytorch_model(Path("checkpoints_raw_ev_ema_s21/best.pt"), dev), "raw_ev_ema")]
GROUPS = (0.0253, 0.038, 0.076)
def wc(p, y):
    p = np.clip(p, -2, 2); y = np.clip(y, -2, 2); w = np.abs(y)
    a = p - np.average(p, weights=w); b = y - np.average(y, weights=w)
    return float((w * a * b).sum() / np.sqrt((w * a * a).sum() * (w * b * b).sum() + 1e-30))
pf = pq.ParquetFile("datasets/valid.parquet"); R = {g: [] for g in range(3)}
with torch.no_grad():
    for rg in np.linspace(0, pf.num_row_groups - 1, 300).astype(int):
        d = pf.read_row_group(int(rg), columns=FEATURE_COLUMNS + ["t0", "t1", "need_prediction", "is_scored"]).to_pandas()
        raw = d[FEATURE_COLUMNS].to_numpy(np.float32); v = raw[:, 48:52]; eq = (v == v[:, :1]).all(1)
        g = int(np.argmin([abs(float(np.median(v[eq, 0])) - x) for x in GROUPS]))
        p = np.mean([m(torch.from_numpy(build_features(raw, fs)).unsqueeze(0).to(dev))[0][0].float().cpu().numpy() for m, fs in ms], 0)
        y = d[["t0", "t1"]].to_numpy(np.float32); n = d.need_prediction.values; s = n & d.is_scored.values
        upd = np.r_[True, np.any(raw[1:, :22] != raw[:-1, :22], axis=1)]
        R[g].append((p[n], y[n], s[n], s.sum(), upd[s].mean() if s.any() else np.nan, upd[n].mean()))
for g in list(range(3)) + ["all"]:
    S = sum(R.values(), []) if g == "all" else R[g]
    P, Y, M = (np.concatenate([x[i] for x in S]) for i in range(3))
    f = lambda m: np.mean([wc(P[m, j], Y[m, j]) for j in range(2)])
    print(f"group {g if g=='all' else GROUPS[g]}: n={len(S)} scored rows/seq mean {np.mean([x[3] for x in S]):.0f} (min {min(x[3] for x in S)}, max {max(x[3] for x in S)}) | "
          f"WP scored {f(M):.4f} | WP all rows {f(np.ones(len(M),bool)):.4f} | WP unscored {f(~M):.4f} | "
          f"|t| scored {np.abs(Y[M]).mean():.3f} unscored {np.abs(Y[~M]).mean():.3f} | update share scored {np.nanmean([x[4] for x in S]):.3f} all {np.mean([x[5] for x in S]):.3f}")
