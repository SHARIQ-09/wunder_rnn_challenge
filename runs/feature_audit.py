"""Audit the 32 engineered features: duplicates, linear redundancy w.r.t. the raw row, and target signal."""
import sys, numpy as np, pyarrow.parquet as pq
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import utils
from src.features import extract_32_features_sequence
NAMES = ["spread0", "disp_b0", "disp_a0", "f4(=spread0)", "ofi0", "imb_L0", "imb_L2", "imb_L4", "imb_L10", "spread1",
         "midgap", "ofi_diff", "dp_sum", "dv_sum", "money_flow", "ret0", "ema_ret.2", "ema_ret.05", "ema_ret.01",
         "ret_div", "ema_ofi.2", "ema_ofi.05", "ema_ofi.01", "ofi_persist", "ema_ret1.05", "leadlag", "ema_midgap",
         "ema_ret2", "ema_dv", "ema_mf", "ema_ret.002", "ofi_shift"]
pf = pq.ParquetFile("datasets/train.parquet"); rng = np.random.default_rng(1)
R, F, Y, M = [], [], [], []
for rg in rng.choice(pf.num_row_groups, 60, replace=False):
    df = pf.read_row_group(int(rg)).to_pandas()
    raw = df[utils.FEATURE_COLUMNS].to_numpy(np.float32); f = extract_32_features_sequence(raw)
    idx = np.flatnonzero(df.need_prediction.values)[::5]
    R.append(raw[idx]); F.append(f[idx, 112:]); Y.append(np.clip(df[["t0", "t1"]].to_numpy(np.float64)[idx], -2, 2))
R, F, Y = (np.vstack(a).astype(np.float64) for a in (R, F, Y))
A = np.hstack([R, np.ones((len(R), 1))])
coef, *_ = np.linalg.lstsq(A, F, rcond=None)
r2 = 1 - ((F - A @ coef) ** 2).sum(0) / ((F - F.mean(0)) ** 2).sum(0)
def wcorr(x, y):
    w = np.abs(y); xm = (w * x).sum() / w.sum(); ym = (w * y).sum() / w.sum()
    return (w * (x - xm) * (y - ym)).sum() / np.sqrt((w * (x - xm) ** 2).sum() * (w * (y - ym) ** 2).sum())
# signal the raw row already gives linearly (upper bound for single features to compare against)
raw_fit = [wcorr(A @ np.linalg.lstsq(A, Y[:, k], rcond=None)[0], Y[:, k]) for k in range(2)]
print(f"rows={len(F)}  linear fit of raw row -> t0 corr {raw_fit[0]:.3f}, t1 corr {raw_fit[1]:.3f}")
C = np.corrcoef(F.T)
print(f"{'#':>3} {'feature':14s} {'R2 raw':>7} {'corr t0':>8} {'corr t1':>8}  most-correlated other")
for i, n in enumerate(NAMES):
    c = C[i].copy(); c[i] = 0; j = int(np.nanargmax(np.abs(c)))
    print(f"{i+1:3d} {n:14s} {r2[i]:7.3f} {wcorr(F[:, i], Y[:, 0]):8.3f} {wcorr(F[:, i], Y[:, 1]):8.3f}  {NAMES[j]} ({c[j]:+.3f})")
