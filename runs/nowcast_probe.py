"""Read-only probe: how much of the target is predictable from the CURRENT row only (no memory)?
Trains LightGBM on rows from train sequences, scores official-mask WP on 150 valid sequences."""
import sys, numpy as np, pyarrow.parquet as pq, lightgbm as lgb
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import utils
from utils import weighted_pearson
from src.features import extract_32_features_sequence
rng = np.random.default_rng(0)
def load(path, rgs, every, scored_only=False):
    pf = pq.ParquetFile(path); X, Y = [], []
    for i in rgs:
        df = pf.read_row_group(i).to_pandas()
        f = extract_32_features_sequence(df[utils.FEATURE_COLUMNS].to_numpy(np.float32))
        m = df.need_prediction.values.copy()
        if scored_only: m &= df.is_scored.values
        idx = np.flatnonzero(m)[::every]
        X.append(f[idx]); Y.append(df[['t0', 't1']].to_numpy(np.float32)[idx])
    return np.vstack(X), np.vstack(Y)
Xtr, Ytr = load('datasets/train.parquet', rng.choice(10607, 400, replace=False), 20)
Xva, Yva = load('datasets/valid.parquet', range(0, 1873, 12), 1, scored_only=True)
print('train rows', len(Xtr), 'valid scored rows', len(Xva))
sc = []
for k in range(2):
    m = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=127, subsample=0.7, subsample_freq=1, colsample_bytree=0.7, verbose=-1)
    w = np.abs(np.clip(Ytr[:, k], -2, 2)) + 0.05
    m.fit(Xtr, np.clip(Ytr[:, k], -2, 2), sample_weight=w)
    sc.append(weighted_pearson(Yva[:, k], m.predict(Xva)))
print(f"current-row-only LightGBM: t0={sc[0]:.4f} t1={sc[1]:.4f} WP={np.mean(sc):.4f}  (pooled over 157 valid seqs, official mask)")
