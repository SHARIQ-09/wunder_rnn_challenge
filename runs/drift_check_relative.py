"""Same adversarial check, but volume/trade/a-columns replaced by x - causal EMA(x) within each sequence.
If AUC drops sharply, a relative representation removes the train/valid shift. Read-only."""
import sys, numpy as np, pyarrow.parquet as pq, lightgbm as lgb
from pathlib import Path
from scipy.signal import lfilter
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import utils
from src.features import extract_32_features_sequence
from sklearn.metrics import roc_auc_score
ALPHA = float(sys.argv[1]) if len(sys.argv) > 1 else 0.01
VOL = list(range(22, 44)) + list(range(74, 96)) + list(range(44, 52)) + list(range(96, 104)) + list(range(104, 112))
def rel(x):
    out = x.copy(); b = np.array([ALPHA]); a = np.array([1.0, -(1 - ALPHA)])
    for j in VOL:
        ema, _ = lfilter(b, a, x[:, j], zi=np.array([(1 - ALPHA) * x[0, j]]))
        out[:, j] = x[:, j] - ema
    return out
rng = np.random.default_rng(0)
def load(path, rgs, label, every=10):
    pf = pq.ParquetFile(path); X, G = [], []
    for i in rgs:
        raw = pf.read_row_group(i, columns=utils.FEATURE_COLUMNS).to_pandas().to_numpy(np.float32)
        f = extract_32_features_sequence(raw)
        f[:, :112] = rel(f[:, :112])
        f = f[99::every]; X.append(f); G.append(np.full(len(f), i if label == 0 else 100000 + i))
    X = np.vstack(X); return X, np.full(len(X), label), np.concatenate(G)
tr = rng.choice(10607, 300, replace=False); va = rng.choice(1873, 300, replace=False)
Xa, ya, ga = load('datasets/train.parquet', tr, 0); Xb, yb, gb = load('datasets/valid.parquet', va, 1)
X = np.vstack([Xa, Xb]); y = np.concatenate([ya, yb]); g = np.concatenate([ga, gb])
ug = np.unique(g); test_g = set(rng.choice(ug, len(ug) // 4, replace=False).tolist())
te = np.array([x in test_g for x in g])
def auc(cols):
    clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, subsample=0.5, subsample_freq=1, colsample_bytree=0.5, verbose=-1)
    clf.fit(X[~te][:, cols], y[~te]); return roc_auc_score(y[te], clf.predict_proba(X[te][:, cols])[:, 1]), clf
allc = list(range(144)); a_all, clf = auc(allc)
print(f"alpha={ALPHA}: AUC with relative volumes, all 144 features = {a_all:.4f}")
names = list(utils.FEATURE_COLUMNS) + [f"eng{j}" for j in range(32)]
imp = clf.booster_.feature_importance('gain'); o = np.argsort(-imp)[:8]
print("  top drift features now:", [(names[j], round(imp[j] / imp.sum(), 3)) for j in o])
a_raw, clf2 = auc(list(range(112)))
print(f"alpha={ALPHA}: AUC with relative volumes, 112 raw columns only = {a_raw:.4f}")
imp = clf2.booster_.feature_importance('gain'); o = np.argsort(-imp)[:8]
print("  top drift features (raw only):", [(names[j], round(imp[j] / imp.sum(), 3)) for j in o])
