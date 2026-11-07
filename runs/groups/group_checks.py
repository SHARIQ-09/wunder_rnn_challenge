"""Per-group checks of the A+B ensemble. Groups = resting value of the i0 dv columns (3 values).
Part 1: score by group, mix sensitivity. Part 2: held-out per-group recalibration. Part 3: why groups differ (incl. train vs valid).
"""
import sys
from pathlib import Path
import numpy as np, pyarrow.parquet as pq, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils import FEATURE_COLUMNS, GlobalAccumulator
from src.evaluate import load_pytorch_model
from src.feature_sets import build_features

dev = torch.device("cuda")
ms = [(load_pytorch_model(Path("checkpoints_norm_ft/best_norm_ft.pt"), dev), "v144"),
      (load_pytorch_model(Path("checkpoints_raw_ev_ema_s21/best.pt"), dev), "raw_ev_ema")]
DV = FEATURE_COLUMNS[48:52]
GROUPS = (0.0253, 0.038, 0.076)


def group_of(raw):
    v = raw[:, 48:52]; eq = (v == v[:, :1]).all(1)
    return int(np.argmin([abs(float(np.median(v[eq, 0])) - g) for g in GROUPS]))


def run(split, rgs, scored):
    pf = pq.ParquetFile(f"datasets/{split}.parquet")
    cols = FEATURE_COLUMNS + ["t0", "t1", "need_prediction"] + (["is_scored"] if scored else [])
    out, acc = [], GlobalAccumulator()
    with torch.no_grad():
        for rg in rgs:
            d = pf.read_row_group(int(rg), columns=cols).to_pandas()
            raw = d[FEATURE_COLUMNS].to_numpy(np.float32)
            y = d[["t0", "t1"]].to_numpy(np.float32)
            m = d.need_prediction.values & (d.is_scored.values if scored else True)
            p = np.mean([mm(torch.from_numpy(build_features(raw, fs)).unsqueeze(0).to(dev))[0][0].float().cpu().numpy() for mm, fs in ms], 0)
            if scored:
                acc.add(y, p, m)
            upd = np.any(raw[1:, :22] != raw[:-1, :22], axis=1).mean()
            out.append(dict(g=group_of(raw), p=p[m], y=y[m], upd=float(upd), chg=float((np.diff(y[m], axis=0) != 0).mean())))
    return out, (acc.result() if scored else None)


def wc(p, y):
    p = np.clip(p, -2, 2); y = np.clip(y, -2, 2); w = np.abs(y)
    a = p - np.average(p, weights=w); b = y - np.average(y, weights=w)
    return float((w * a * b).sum() / np.sqrt((w * a * a).sum() * (w * b * b).sum() + 1e-30))


def wp(S, P=None):
    Pc = np.concatenate([s["p"] for s in S]) if P is None else P
    Yc = np.concatenate([s["y"] for s in S])
    return np.mean([wc(Pc[:, j], Yc[:, j]) for j in range(2)])


def t_wp(S, j):
    return wc(np.concatenate([s["p"][:, j] for s in S]), np.concatenate([s["y"][:, j] for s in S]))


n_val = pq.ParquetFile("datasets/valid.parquet").num_row_groups
V, off = run("valid", range(n_val), True)
print(f"valid {len(V)} seqs. official WP {off['weighted_pearson']:.6f}; this script's wc {wp(V):.6f}", flush=True)
n_tr = pq.ParquetFile("datasets/train.parquet").num_row_groups
T, _ = run("train", np.linspace(0, n_tr - 1, 150).astype(int), False)

print("\n== 1. Score by group ==")
for g, name in enumerate(GROUPS):
    S = [s for s in V if s["g"] == g]; R = [s for s in T if s["g"] == g]
    Wt = sum(np.abs(s["y"]).sum() for s in S) / sum(np.abs(s["y"]).sum() for s in V)
    seq = [wp([s]) for s in S]
    print(f"rest={name}: valid n={len(S)} share={len(S)/len(V):.3f} metric-weight={Wt:.3f} WP={wp(S):.4f} (t0 {t_wp(S,0):.4f} t1 {t_wp(S,1):.4f}) "
          f"per-seq p10/50/90 {np.round(np.percentile(seq,[10,50,90]),3)} | train-sample n={len(R)} WP={wp(R):.4f}")
print(f"train-sample all groups WP={wp(T):.4f}")

print("\n== Mix sensitivity: global WP if groups had these sequence shares (bootstrap-free reweighting by resampling) ==")
rng = np.random.default_rng(0)
by = [[s for s in V if s["g"] == g] for g in range(3)]
for mix in [(.59,.22,.20),(.57,.34,.09),(.8,.1,.1),(.9,.05,.05),(1,0,0)]:
    vals = []
    for _ in range(12):
        S = []
        for g, sh in enumerate(mix):
            k = int(round(sh * 400))
            if k: S += [by[g][i] for i in rng.integers(0, len(by[g]), k)]
        vals.append(wp(S))
    print(f"mix {mix}: WP {np.mean(vals):.4f} +- {np.std(vals):.4f}")

print("\n== 2. Held-out per-group recalibration (slope+intercept per group per target, fit on one half, scored on the other) ==")
idx = np.arange(len(V)); half = [idx[idx % 2 == 0], idx[idx % 2 == 1]]
for a, b in [(0, 1), (1, 0)]:
    tr, te = [V[i] for i in half[a]], [V[i] for i in half[b]]
    Pn_slope, Pn_full, Pn_glob = [], [], []
    coef = {}
    for g in range(3):
        for j in range(2):
            S = [s for s in tr if s["g"] == g]
            p = np.concatenate([s["p"][:, j] for s in S]); y = np.clip(np.concatenate([s["y"][:, j] for s in S]), -2, 2); w = np.abs(y)
            sl = (w * p * y).sum() / (w * p * p).sum()
            X = np.column_stack([p, np.ones_like(p)]); c = np.linalg.solve((X * w[:, None]).T @ X, (X * w[:, None]).T @ y)
            coef[g, j] = (sl, c[0], c[1])
    for s in te:
        q1, q2 = s["p"].copy(), s["p"].copy()
        for j in range(2):
            sl, c0, c1 = coef[s["g"], j]
            q1[:, j] = s["p"][:, j] * sl; q2[:, j] = s["p"][:, j] * c0 + c1
        Pn_slope.append(q1); Pn_full.append(q2)
    print(f"fit half {a}: base {wp(te):.5f} | per-group slope {wp(te, np.concatenate(Pn_slope)):.5f} | slope+intercept {wp(te, np.concatenate(Pn_full)):.5f}")
    print("   coef (slope t0,t1 by group):", [tuple(round(coef[g, j][0], 3) for j in range(2)) for g in range(3)],
          "intercepts:", [tuple(round(coef[g, j][2], 3) for j in range(2)) for g in range(3)])

print("\n== 3. Why groups differ ==")
for g, name in enumerate(GROUPS):
    for lab, D in (("valid", V), ("train", T)):
        S = [s for s in D if s["g"] == g]
        y = np.concatenate([s["y"] for s in S]); p = np.concatenate([s["p"] for s in S])
        print(f"rest={name} {lab}: |t| mean {np.abs(y).mean():.3f} p50 {np.median(np.abs(y)):.3f} p99 {np.percentile(np.abs(y),99):.2f} std t {y.std():.3f} "
              f"std pred {p.std():.3f} distinct t0 {len(np.unique(y[:,0]))} corr(t0,t1) {np.corrcoef(y[:,0],y[:,1])[0,1]:.3f} target-change rate {np.mean([s['chg'] for s in S]):.3f} update rate {np.mean([s['upd'] for s in S]):.3f}")
    S = [s for s in V if s["g"] == g]
    print(f"   within-group corr(per-seq WP, update rate) {np.corrcoef([wp([s]) for s in S],[s['upd'] for s in S])[0,1]:.3f}")
np.savez("runs/groups/valid_groups.npz", g=np.array([s["g"] for s in V]), upd=np.array([s["upd"] for s in V]))
