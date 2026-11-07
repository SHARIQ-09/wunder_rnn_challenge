"""LightGBM screen: slim base (459) vs + each pressure group vs + all (100 train seqs -> 80 valid seqs, official mask)."""
import sys
from pathlib import Path
import numpy as np, pyarrow.parquet as pq, lightgbm as lgb
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils import FEATURE_COLUMNS
from src.feature_sets import build_features, PRESSURE_GROUPS, N_EV_EMA

def load(pf, ids, step, scored):
    X, Y = [], []
    for i in ids:
        d = pf.read_row_group(int(i), columns=FEATURE_COLUMNS + ['t0', 't1', 'need_prediction'] + (['is_scored'] if scored else [])).to_pandas()
        F = build_features(d[FEATURE_COLUMNS].to_numpy(np.float32), 'pressure')
        m = d.need_prediction.values & (d.is_scored.values if scored else True)
        idx = np.flatnonzero(m)[::step]; X.append(F[idx]); Y.append(d[['t0', 't1']].to_numpy(np.float32)[idx])
    return np.vstack(X), np.clip(np.vstack(Y), -2, 2)

def wp(y, p):
    w = np.abs(y); my, mp = np.average(y, weights=w), np.average(p, weights=w)
    return np.sum(w*(y-my)*(p-mp)) / np.sqrt(np.sum(w*(y-my)**2) * np.sum(w*(p-mp)**2))

rng = np.random.default_rng(5)
TR = pq.ParquetFile('datasets/train.parquet'); VA = pq.ParquetFile('datasets/valid.parquet')
Xtr, Ytr = load(TR, rng.choice(TR.num_row_groups, 100, replace=False), 10, False)
Xva, Yva = load(VA, rng.choice(VA.num_row_groups, 80, replace=False), 2, True)
print('train rows', len(Xtr), 'valid scored rows', len(Xva), '| finite:', np.isfinite(Xtr).all(), np.isfinite(Xva).all(), flush=True)
P0 = 112 + N_EV_EMA + 3
names = [f'{g}{k}' for g, (a, b) in PRESSURE_GROUPS.items() for k in range(b - a)]
print('new-feature ranges (train): min / max')
print(np.round(np.c_[Xtr[:, P0:].min(0), Xtr[:, P0:].max(0)].T, 2))

models = {}
def run(cols, tag):
    s = []
    for j in range(2):
        m = lgb.LGBMRegressor(n_estimators=300, learning_rate=.05, num_leaves=63, min_child_samples=200, subsample=.8, subsample_freq=1, colsample_bytree=.5, verbose=-1)
        m.fit(Xtr[:, cols], Ytr[:, j], sample_weight=np.abs(Ytr[:, j]) + 1e-3); s.append(wp(Yva[:, j], m.predict(Xva[:, cols]))); models[(tag, j)] = m
    print(f'{tag:28s} t0 {s[0]:.4f}  t1 {s[1]:.4f}  WP {np.mean(s):.4f}', flush=True)
    return np.mean(s)

EVE = list(range(112 + N_EV_EMA)); SLIM = list(range(P0))
run(EVE, 'raw_ev_ema 456'); base = run(SLIM, 'slim 459')
for g, (a, b) in PRESSURE_GROUPS.items():
    run(SLIM + list(range(P0 + a, P0 + b)), f'slim + {g}')
run(list(range(Xtr.shape[1])), 'slim + ALL (pressure set)')
imp = sum(models[('slim + ALL (pressure set)', j)].booster_.feature_importance('gain') for j in range(2))
new = imp[P0:]; order = np.argsort(new)[::-1]
print('top new features by gain share of all gain:')
for k in order[:15]: print(f'  {names[k]:14s} {new[k] / imp.sum():.4f}')
print(f'new block total gain share: {new.sum() / imp.sum():.3f}')
