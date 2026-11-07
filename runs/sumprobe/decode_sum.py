"""Can t0+t1 be decoded from the inputs? LightGBM (raw_ev_ema, own data only), trained on train, tested on valid.
Decision metric: held-out WP after swapping A+B's sum for a blend with the decoded sum (t0-t1 kept)."""
import sys
from pathlib import Path
import numpy as np, pyarrow.parquet as pq, torch, lightgbm as lgb
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils import FEATURE_COLUMNS as F
from src.evaluate import load_pytorch_model
from src.feature_sets import build_features
dev = torch.device("cuda")
ms = [(load_pytorch_model(Path("checkpoints_norm_ft/best_norm_ft.pt"), dev), "v144"), (load_pytorch_model(Path("checkpoints_raw_ev_ema_s21/best.pt"), dev), "raw_ev_ema")]
def wc(p, y):
    p = np.clip(p, -2, 2); y = np.clip(y, -2, 2); w = np.abs(y)
    a = p - np.average(p, weights=w); b = y - np.average(y, weights=w)
    return float((w * a * b).sum() / np.sqrt((w * a * a).sum() * (w * b * b).sum() + 1e-30))
wp = lambda P, Y: np.mean([wc(P[:, j], Y[:, j]) for j in range(2)])
Gt, Gv = np.load("runs/groups/train_group.npy"), np.load("runs/groups/valid_group.npy")
pt, pv = pq.ParquetFile("datasets/train.parquet"), pq.ParquetFile("datasets/valid.parquet")
cols = F + ["t0", "t1", "need_prediction"]

print("== Structure of t0+t1 (train, 200 seqs, all required rows) ==", flush=True)
X, S, GG, upd_rates = [], [], [], []
for rg in np.linspace(0, pt.num_row_groups - 1, 200).astype(int):
    d = pt.read_row_group(int(rg), columns=cols).to_pandas(); n = d.need_prediction.values
    raw = d[F].to_numpy(np.float32); s = (d.t0.values + d.t1.values).astype(np.float64)
    upd = np.r_[True, np.any(raw[1:, :22] != raw[:-1, :22], axis=1)]; ch = np.r_[False, s[1:] != s[:-1]]
    upd_rates.append((ch.mean(), (ch & upd).sum() / max(ch.sum(), 1)))
    S.append(s[n][::5]); GG.append(np.full(len(S[-1]), Gt[rg])); X.append(build_features(raw, "raw_ev_ema")[n][::5])
X, S, GG = np.vstack(X), np.concatenate(S), np.concatenate(GG)
for k in range(3):
    v = S[GG == k]; u = np.unique(np.round(v, 4)); steps = np.diff(u)
    print(f"group {k}: mean {v.mean():+.3f} std {v.std():.3f} share>0 {(v > 1e-6).mean():.4f} share=0 {(np.abs(v) <= 1e-6).mean():.4f} distinct {len(u)} smallest steps {np.round(np.sort(steps)[:3], 4)} median step {np.median(steps):.4f}")
print(f"sum changes on {np.mean([x[0] for x in upd_rates]):.3f} of rows; of those, {np.mean([x[1] for x in upd_rates]):.3f} are on i0 price updates", flush=True)

tr = np.random.default_rng(0).permutation(len(S)); cut = int(.9 * len(S))
m = lgb.train(dict(objective="regression", learning_rate=.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=.3, bagging_fraction=.7, bagging_freq=1, verbose=-1, num_threads=6),
              lgb.Dataset(X[tr[:cut]], S[tr[:cut]]), 400, valid_sets=[lgb.Dataset(X[tr[cut:]], S[tr[cut:]])], callbacks=[lgb.early_stopping(30, verbose=False)])
print("LightGBM rounds", m.best_iteration, flush=True)
imp = m.feature_importance("gain"); o = np.argsort(-imp)[:10]; print("top features (index in raw_ev_ema):", o.tolist(), np.round(imp[o] / imp.sum(), 3).tolist())
del X

print("\n== Valid ==", flush=True)
P, Y, D, G = [], [], [], []
vcols = cols + ["is_scored"]
with torch.no_grad():
    for k, rg in enumerate(np.linspace(0, pv.num_row_groups - 1, 240).astype(int)):
        d = pv.read_row_group(int(rg), columns=vcols).to_pandas(); raw = d[F].to_numpy(np.float32); mk = d.need_prediction.values & d.is_scored.values
        if not mk.any(): continue
        p = np.mean([mm(torch.from_numpy(build_features(raw, fs)).unsqueeze(0).to(dev))[0][0].float().cpu().numpy() for mm, fs in ms], 0)
        P.append(p[mk]); Y.append(d[["t0", "t1"]].to_numpy(np.float64)[mk]); D.append(m.predict(build_features(raw, "raw_ev_ema")[mk])); G.append(np.full(mk.sum(), k))
P, Y, D, G = map(np.concatenate, (P, Y, D, G)); S = Y.sum(1); Ps = P.sum(1); gr = Gv[np.linspace(0, pv.num_row_groups - 1, 240).astype(int)][G]
for nm, sel in [("all", np.ones(len(S), bool))] + [(f"group {k}", gr == k) for k in range(3)]:
    r2 = lambda q: 1 - ((S[sel] - q[sel]) ** 2).sum() / ((S[sel] - S[sel].mean()) ** 2).sum()
    print(f"{nm}: corr with true sum: A+B {np.corrcoef(Ps[sel], S[sel])[0,1]:.4f}  decoded {np.corrcoef(D[sel], S[sel])[0,1]:.4f} | R2 A+B {r2(Ps):.4f} decoded {r2(D):.4f}")
print("\nHeld-out blend (a, b, c fitted on one half of sequences: sum' = a*A+B sum + b*decoded + c):")
for par in (0, 1):
    a_, b_ = G % 2 == par, G % 2 != par
    A = np.column_stack([Ps[a_], D[a_], np.ones(a_.sum())]); w = np.abs(np.clip(Y[a_], -2, 2)).sum(1)
    c = np.linalg.solve((A * w[:, None]).T @ A, (A * w[:, None]).T @ S[a_])
    ns = np.column_stack([Ps[b_], D[b_], np.ones(b_.sum())]) @ c; d = P[b_, 0] - P[b_, 1]
    Q = np.column_stack([(d + ns) / 2, (ns - d) / 2]); O = np.column_stack([(d + S[b_]) / 2, (S[b_] - d) / 2])
    print(f"  coef {np.round(c,3)}: base {wp(P[b_], Y[b_]):.5f} -> blended {wp(Q, Y[b_]):.5f} (oracle sum {wp(O, Y[b_]):.5f})")
