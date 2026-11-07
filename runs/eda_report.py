"""Read-only EDA for the report: targets, regimes, scoring mask, feature behaviour, train vs valid."""
import sys, numpy as np, pyarrow.parquet as pq
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import utils
C = utils.FEATURE_COLUMNS
rng = np.random.default_rng(7)
tr, va = pq.ParquetFile("datasets/train.parquet"), pq.ParquetFile("datasets/valid.parquet")

def seqs(pf, rgs):
    for i in rgs:
        yield pf.read_row_group(int(i)).to_pandas()

print("== 1. Targets (train, 150 random seqs, prediction rows)")
Y, seq_std, seq_mean, raw_means, raw_wstd, raw_all = [], [], [], [], [], []
upd = {k: [] for k in ["i0_p", "i0_v", "i0_dp", "i0_dv", "i1_p", "i1_v", "i1_dp", "i1_dv", "a"]}
blocks = {"i0_p": (0, 22), "i0_v": (22, 44), "i0_dp": (44, 48), "i0_dv": (48, 52), "i1_p": (52, 74),
          "i1_v": (74, 96), "i1_dp": (96, 100), "i1_dv": (100, 104), "a": (104, 112)}
askgtbid = []
for df in seqs(tr, rng.choice(tr.num_row_groups, 150, replace=False)):
    m = df.need_prediction.values
    y = df[["t0", "t1"]].to_numpy(np.float64)[m]; Y.append(y)
    seq_std.append(y.std(0)); seq_mean.append(y.mean(0))
    x = df[C].to_numpy(np.float64)
    raw_means.append(x.mean(0)); raw_wstd.append(x.std(0)); raw_all.append(x[::50])
    for k, (a, b) in blocks.items():
        upd[k].append(np.mean(np.any(x[1:, a:b] != x[:-1, a:b], axis=1)))
    askgtbid.append(np.mean(x[:, 11:22].mean(1) > x[:, 0:11].mean(1)))
Y = np.vstack(Y); seq_std = np.array(seq_std); seq_mean = np.array(seq_mean)
for k, n in enumerate(["t0", "t1"]):
    y = Y[:, k]; q = np.quantile(y, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    kurt = ((y - y.mean()) ** 4).mean() / y.var() ** 2
    yc = np.clip(y, -2, 2); w = np.abs(yc); ws = np.sort(w)[::-1]
    print(f"{n}: mean={y.mean():+.4f} std={y.std():.4f} kurtosis={kurt:.1f}  quantiles 1/5/25/50/75/95/99% = {np.round(q, 3)}")
    print(f"    |t|>2 (clipped): {np.mean(np.abs(y) > 2) * 100:.2f}%   exact zeros: {np.mean(y == 0) * 100:.2f}%   "
          f"top 10% rows carry {ws[:len(ws) // 10].sum() / ws.sum() * 100:.1f}% of metric weight, top 1% carry {ws[:len(ws) // 100].sum() / ws.sum() * 100:.1f}%")
    print(f"    per-sequence std: min={seq_std[:, k].min():.3f} median={np.median(seq_std[:, k]):.3f} max={seq_std[:, k].max():.3f};"
          f"  per-sequence mean: sd across seqs={seq_mean[:, k].std():.4f}")
print(f"corr(t0,t1) pooled = {np.corrcoef(Y[:, 0], Y[:, 1])[0, 1]:.3f}")

print("\n== 2. Features")
raw_means = np.array(raw_means); raw_wstd = np.array(raw_wstd); raw_all = np.vstack(raw_all)
for k, (a, b) in blocks.items():
    tot = raw_all[:, a:b].var(0); between = raw_means[:, a:b].var(0)
    print(f"{k:6s} cols {a:3d}-{b - 1:3d}: update freq {np.mean(upd[k]) * 100:5.1f}% of steps;  share of variance BETWEEN sequences "
          f"{np.median(between / tot) * 100:5.1f}% (median over cols); typical within-seq std {np.median(raw_wstd[:, a:b]):.3f}")
print(f"mean(ask price-like) > mean(bid price-like) on {np.mean(askgtbid) * 100:.1f}% of rows (i0)")
cc = np.corrcoef(raw_all[:, 0:22].T); print(f"i0 price-like: mean |corr| between columns = {np.abs(cc[np.triu_indices(22, 1)]).mean():.3f}")
ev = np.linalg.eigvalsh(np.cov(raw_all[:, 0:22].T))[::-1]; print(f"i0 price-like: top-3 principal components explain {np.round(ev[:3] / ev.sum() * 100, 1)} % of variance")
ev = np.linalg.eigvalsh(np.cov(raw_all[:, 22:44].T))[::-1]; print(f"i0 volume-like: top-3 principal components explain {np.round(ev[:3] / ev.sum() * 100, 1)} % of variance")

print("\n== 3. Scoring mask (valid, all 1873 seqs)")
frac, runs_, gaps, ys, yu, first = [], [], [], [], [], []
for df in seqs(va, range(va.num_row_groups)):
    s = (df.is_scored.values & df.need_prediction.values)
    frac.append(s.mean())
    if s.any():
        d = np.diff(np.concatenate([[0], s.astype(int), [0]]))
        st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        runs_ += (en - st).tolist(); gaps += (st[1:] - en[:-1]).tolist(); first.append(st[0])
    if len(ys) < 300:
        y = np.abs(df[["t0", "t1"]].to_numpy(np.float64)); m = df.need_prediction.values
        ys.append(y[s]); yu.append(y[m & ~s])
frac = np.array(frac); runs_ = np.array(runs_); gaps = np.array(gaps)
print(f"scored fraction of all rows: mean {frac.mean() * 100:.2f}%; per-seq quantiles 0/10/50/90/100% = {np.round(np.quantile(frac, [0, .1, .5, .9, 1]) * 100, 2)}%; seqs with no scored rows: {np.sum(frac == 0)}")
print(f"scored runs: count {len(runs_)}, length quantiles 10/50/90/99% = {np.quantile(runs_, [.1, .5, .9, .99])}, single-row runs {np.mean(runs_ == 1) * 100:.1f}%")
print(f"gaps between runs: quantiles 10/50/90% = {np.quantile(gaps, [.1, .5, .9])}; first scored step median {np.median(first):.0f}")
ys = np.vstack(ys); yu = np.vstack(yu)
print(f"mean |t| on scored rows t0={ys[:, 0].mean():.3f} t1={ys[:, 1].mean():.3f}  vs unscored t0={yu[:, 0].mean():.3f} t1={yu[:, 1].mean():.3f}")

print("\n== 4. Train vs valid targets")
Yv = np.vstack([df[["t0", "t1"]].to_numpy(np.float64)[df.need_prediction.values] for df in seqs(va, rng.choice(va.num_row_groups, 150, replace=False))])
print(f"train std t0/t1 = {Y.std(0).round(4)}   valid std t0/t1 = {Yv.std(0).round(4)}")
