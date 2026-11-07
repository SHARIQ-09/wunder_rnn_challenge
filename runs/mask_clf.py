import sys, numpy as np, pyarrow.parquet as pq, lightgbm as lgb
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import utils
from src.features import extract_32_features_sequence
from sklearn.metrics import roc_auc_score, average_precision_score
pf = pq.ParquetFile('datasets/valid.parquet')
def load(rgs):
    X, S = [], []
    for i in rgs:
        df = pf.read_row_group(i).to_pandas()
        raw = df[utils.FEATURE_COLUMNS].to_numpy(np.float32)
        f = extract_32_features_sequence(raw)
        d1 = np.vstack([np.zeros((1, 112), np.float32), np.diff(raw, axis=0)])  # 1-step changes
        st = df.step_in_seq.values[:, None].astype(np.float32)
        m = df.need_prediction.values
        X.append(np.hstack([f, d1, st])[m]); S.append(df.is_scored.values[m])
    return np.vstack(X), np.concatenate(S)
rng = np.random.default_rng(0)
Xtr, Str = load(rng.choice(1400, 120, replace=False))
Xte, Ste = load(range(1400, 1873, 12))
clf = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63, subsample=0.5, subsample_freq=1, colsample_bytree=0.5, verbose=-1)
clf.fit(Xtr, Str)
p = clf.predict_proba(Xte)[:, 1]
print('AUC', round(roc_auc_score(Ste, p), 4), 'AP', round(average_precision_score(Ste, p), 4), 'base rate', round(Ste.mean(), 4))
imp = np.argsort(-clf.feature_importances_)[:15]
names = [f'f{j}' if j < 144 else (f'd_{utils.FEATURE_COLUMNS[j-144]}' if j < 256 else 'step') for j in imp]
print('top', list(zip(names, clf.feature_importances_[imp].tolist())))
