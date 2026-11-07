"""Read-only: does a LightGBM row model (different model family) add to the GRU ensemble?
LightGBM trains on rows from 400 train sequences; blends are scored on every 12th valid sequence (official mask)."""
import sys, numpy as np, torch, pyarrow.parquet as pq, lightgbm as lgb
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import GlobalAccumulator, FEATURE_COLUMNS, TARGET_COLUMNS
from src.feature_sets import build_features
from src.evaluate import load_pytorch_model
FS = "v144_ev_ema"
rng = np.random.default_rng(0)
tr = pq.ParquetFile("datasets/train.parquet")
X, Y = [], []
for i in rng.choice(tr.num_row_groups, 400, replace=False):
    df = tr.read_row_group(int(i)).to_pandas()
    f = build_features(df[FEATURE_COLUMNS].to_numpy(np.float32), FS)
    idx = np.flatnonzero(df.need_prediction.values)[::20]
    X.append(f[idx]); Y.append(df[list(TARGET_COLUMNS)].to_numpy(np.float32)[idx])
X, Y = np.vstack(X), np.clip(np.vstack(Y), -2, 2)
print("train rows", X.shape, flush=True)
gbms = []
for k in range(2):
    m = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=127, subsample=0.7, subsample_freq=1,
                          colsample_bytree=0.5, verbose=-1, n_jobs=6)
    m.fit(X, Y[:, k], sample_weight=np.abs(Y[:, k]) + 0.05); gbms.append(m)
del X, Y
dev = torch.device("cuda")
ckpts = ["checkpoints_norm_ft/best_norm_ft.pt", "checkpoints_raw_ev_ema_s21/best.pt"]
ms = [load_pytorch_model(Path(c), dev) for c in ckpts]
fsets = [torch.load(c, map_location="cpu").get("model_config", {}).get("feature_set", "v144") for c in ckpts]
W = [0.0, 0.1, 0.2, 0.3, 0.5, 1.0]
accs = {w: GlobalAccumulator() for w in W}
va = pq.ParquetFile("datasets/valid.parquet")
with torch.no_grad():
    for rg in range(0, va.num_row_groups, 12):
        df = va.read_row_group(rg).to_pandas()
        raw = df[FEATURE_COLUMNS].to_numpy(np.float32)
        Xs = {f: build_features(raw, f) for f in set(fsets) | {FS}}
        G = np.mean([m(torch.from_numpy(Xs[f]).unsqueeze(0).to(dev))[0][0].float().cpu().numpy() for m, f in zip(ms, fsets)], axis=0)
        B = np.zeros_like(G); need = df.need_prediction.values
        B[need] = np.column_stack([g.predict(Xs[FS][need]) for g in gbms])
        y = df[list(TARGET_COLUMNS)].to_numpy(np.float32); mk = need & df.is_scored.values
        for w in W:
            accs[w].add(y, ((1 - w) * G + w * B).astype(np.float32), mk)
for w in W:
    r = accs[w].result(); print(f"LightGBM weight {w:.1f}: WP={r['weighted_pearson']:.6f} t0={r['t0']:.6f} t1={r['t1']:.6f}")
