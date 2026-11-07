"""Read-only probe: current row + multi-scale causal EMAs of all 112 raw inputs (deviation from EMA).
Same protocol as nowcast_probe.py so numbers are comparable."""
import sys, numpy as np, pyarrow.parquet as pq, lightgbm as lgb
from pathlib import Path
from scipy.signal import lfilter
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import utils
from utils import weighted_pearson
from src.features import extract_32_features_sequence
ALPHAS = [0.1, 0.01, 0.001]
def ema(x, a):
    return lfilter([a], [1, -(1 - a)], x, axis=0, zi=((1 - a) * x[:1]))[0]
def feats(raw):
    f = extract_32_features_sequence(raw)
    s = f[:, :112]  # stationary raw block
    hist = [s - ema(s, a) for a in ALPHAS]
    return np.hstack([f] + hist).astype(np.float32)
rng = np.random.default_rng(0)
def load(path, rgs, every, scored_only=False):
    pf = pq.ParquetFile(path); X, Y = [], []
    for i in rgs:
        df = pf.read_row_group(i).to_pandas()
        f = feats(df[utils.FEATURE_COLUMNS].to_numpy(np.float64))
        m = df.need_prediction.values.copy()
        if scored_only: m &= df.is_scored.values
        idx = np.flatnonzero(m)[::every]
        X.append(f[idx]); Y.append(df[['t0', 't1']].to_numpy(np.float32)[idx])
    return np.vstack(X), np.vstack(Y)
Xtr, Ytr = load('datasets/train.parquet', rng.choice(10607, 400, replace=False), 20)
Xva, Yva = load('datasets/valid.parquet', range(0, 1873, 12), 1, scored_only=True)
print('features', Xtr.shape[1])
sc = []; models = []
for k in range(2):
    m = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=127, subsample=0.7, subsample_freq=1, colsample_bytree=0.5, verbose=-1)
    m.fit(Xtr, np.clip(Ytr[:, k], -2, 2), sample_weight=np.abs(np.clip(Ytr[:, k], -2, 2)) + 0.05)
    sc.append(weighted_pearson(Yva[:, k], m.predict(Xva))); models.append(m)
print(f"current row + EMA history LightGBM: t0={sc[0]:.4f} t1={sc[1]:.4f} WP={np.mean(sc):.4f}")
imp = sum(m.booster_.feature_importance('gain') for m in models)
blocks = {'raw144': imp[:144].sum(), **{f'dev_ema{a}': imp[144 + 112 * i:144 + 112 * (i + 1)].sum() for i, a in enumerate(ALPHAS)}}
tot = sum(blocks.values()); print('gain share by block:', {k: round(v / tot, 3) for k, v in blocks.items()})
