"""Adversarial validation: can a classifier tell train rows from valid rows?
High AUC => distribution shift; top features show what drifts. Read-only."""
import sys, numpy as np, pyarrow.parquet as pq, lightgbm as lgb
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import utils
from src.features import extract_32_features_sequence
from sklearn.metrics import roc_auc_score
rng = np.random.default_rng(0)
def load(path, rgs, label, every=10):
    pf = pq.ParquetFile(path); X, G = [], []
    for i in rgs:
        df = pf.read_row_group(i, columns=utils.FEATURE_COLUMNS).to_pandas()
        f = extract_32_features_sequence(df.to_numpy(np.float32))[99::every]
        X.append(f); G.append(np.full(len(f), i if label == 0 else 100000 + i))
    X = np.vstack(X); return X, np.full(len(X), label), np.concatenate(G)
tr = rng.choice(10607, 300, replace=False); va = rng.choice(1873, 300, replace=False)
Xa, ya, ga = load('datasets/train.parquet', tr, 0); Xb, yb, gb = load('datasets/valid.parquet', va, 1)
X = np.vstack([Xa, Xb]); y = np.concatenate([ya, yb]); g = np.concatenate([ga, gb])
# split by sequence so rows of one sequence never sit on both sides
ug = np.unique(g); test_g = set(rng.choice(ug, len(ug) // 4, replace=False).tolist())
te = np.array([x in test_g for x in g])
clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, subsample=0.5, subsample_freq=1, colsample_bytree=0.5, verbose=-1)
clf.fit(X[~te], y[~te]); p = clf.predict_proba(X[te])[:, 1]
print(f"rows train={len(Xa)} valid={len(Xb)}  sequence-split AUC (train vs valid) = {roc_auc_score(y[te], p):.4f}")
names = list(utils.FEATURE_COLUMNS) + [f"eng{j}" for j in range(32)]
imp = clf.booster_.feature_importance('gain'); o = np.argsort(-imp)[:15]
tot = imp.sum()
for j in o:
    print(f"  {names[j]:8s} gain_share={imp[j]/tot:.3f}  train_mean={Xa[:, j].mean():+.4f} valid_mean={Xb[:, j].mean():+.4f}  train_std={Xa[:, j].std():.4f} valid_std={Xb[:, j].std():.4f}")
