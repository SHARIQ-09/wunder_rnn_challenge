"""Step 3: what are the targets in terms of i0 prices? Per target-unit population, ridge fitted on train seqs, scored on valid seqs.
(a) determinism of sum/diff given the exact price row, (b) innovation: delta target vs same-row delta price at updates,
(c) horizon scan: target vs price change over future and past windows, in steps and in price updates."""
import sys
from pathlib import Path
import numpy as np, pyarrow.parquet as pq
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils import FEATURE_COLUMNS as F
UNITS = (0.6569, 0.2413, 0.46855, 0.12543); NT, NV, SUB = 60, 40, 4
STEPS = (1, 5, 20, 100, 500, 2000); UPDS = (1, 2, 5, 20, 100)
NAMES = ("t0", "t1", "sum", "diff")


def load(split, k, n):
    pf = pq.ParquetFile(f"datasets/{split}.parquet"); ids = np.flatnonzero(np.load(f"runs/decode/{split}_unit.npy") == k)
    out = []
    for rg in np.random.default_rng(k).choice(ids, min(n, len(ids)), replace=False):
        d = pf.read_row_group(int(rg), columns=F[:22] + ["t0", "t1", "need_prediction"]).to_pandas()
        y = d[["t0", "t1"]].to_numpy(np.float64) / UNITS[k]
        out.append((d[F[:22]].to_numpy(np.float64), np.column_stack([y[:, 0], y[:, 1], y.sum(1), y[:, 0] - y[:, 1]]), d.need_prediction.values))
    return out


def fit_score(Xa, Ya, Xb, Yb):
    """Ridge with intercept; returns held-out R2 and corr per target."""
    mx, my = Xa.mean(0), Ya.mean(0); A = Xa - mx
    b = np.linalg.solve(A.T @ A + 1e-3 * len(A) * np.eye(A.shape[1]) * A.var(0).mean(), A.T @ (Ya - my))
    p = (Xb - mx) @ b + my
    r2 = 1 - ((Yb - p) ** 2).sum(0) / ((Yb - Yb.mean(0)) ** 2).sum(0)
    c = [np.corrcoef(p[:, j], Yb[:, j])[0, 1] if p[:, j].std() > 0 else 0 for j in range(Yb.shape[1])]
    return r2, np.array(c)


def show(tag, r):
    print(f"  {tag:14s} " + " ".join(f"{NAMES[j]} R2 {r[0][j]:+.3f} c {r[1][j]:+.3f} |" for j in range(4)), flush=True)


def window(seqs, h, update_clock):
    """X = price(t+h) - price(t), Y = target(t), on sampled required rows. h may be negative (past); h = 0 gives the price row itself minus its row mean."""
    X, Y = [], []
    for p, y, n in seqs:
        if h == 0:
            x = p - p.mean(1, keepdims=True); idx = np.flatnonzero(n)
        elif update_clock:
            up = np.flatnonzero(np.r_[True, np.any(p[1:] != p[:-1], axis=1)])
            pos = np.searchsorted(up, np.arange(len(p)), side="right") - 1      # latest update at or before each row
            tgt = pos + h; ok = (tgt >= 0) & (tgt < len(up)) & n
            idx = np.flatnonzero(ok); x = p[up[tgt[idx]]] - p[idx]
            X.append(x[::SUB]); Y.append(y[idx][::SUB]); continue
        else:
            idx = np.flatnonzero(n & (np.arange(len(p)) + h >= 0) & (np.arange(len(p)) + h < len(p))); x = p[idx + h] - p[idx]
            X.append(x[::SUB]); Y.append(y[idx][::SUB]); continue
        X.append(x[idx][::SUB]); Y.append(y[idx][::SUB])
    return np.vstack(X), np.vstack(Y)


for k, u in enumerate(UNITS):
    tr, va = load("train", k, NT), load("valid", k, NV)
    print(f"\n===== unit {u:.4f} (targets in units; {len(tr)} train, {len(va)} valid seqs) =====")
    # (a) determinism on valid
    for j in (2, 3):
        agree = tot = dist = 0
        for p, y, n in va:
            seen = {}
            for i in np.flatnonzero(n):
                key = p[i].tobytes()
                if key in seen:
                    tot += 1; agree += abs(seen[key] - y[i, j]) < 1e-3
                else:
                    seen[key] = y[i, j]; dist += 1
        print(f"  (a) same price row -> same {NAMES[j]}: {agree / max(tot, 1):.4f} ({dist:,} distinct rows, {tot:,} repeats)")
    # (b) innovation at updates
    def innov(seqs, lag):
        X, Y = [], []
        for p, y, n in seqs:
            d = np.diff(p, axis=0); up = np.flatnonzero(np.any(d != 0, axis=1)); up = up[(up + lag >= 0) & (up + 1 < len(p))]
            ups = up[n[up + 1]]
            if lag == 0:
                X.append(d[ups]); Y.append(y[ups + 1] - y[ups])
            else:                                                            # previous update's price move -> this update's target move
                X.append(d[ups[:-1]]); Y.append(y[ups[1:] + 1] - y[ups[1:]])
        return np.vstack(X), np.vstack(Y)
    Xa, Ya = innov(tr, 0); Xb, Yb = innov(va, 0); print(f"  (b) changed target at updates: {(np.abs(Yb[:, 3]) > 1e-3).mean():.3f}"); show("same update", fit_score(Xa, Ya, Xb, Yb))
    Xa, Ya = innov(tr, 1); Xb, Yb = innov(va, 1); show("prev update", fit_score(Xa, Ya, Xb, Yb))
    # (c) horizons
    show("row level", fit_score(*window(tr, 0, False), *window(va, 0, False)))
    for h in STEPS:
        show(f"future {h} st", fit_score(*window(tr, h, False), *window(va, h, False)))
    for h in STEPS:
        show(f"past {h} st", fit_score(*window(tr, -h, False), *window(va, -h, False)))
    for h in UPDS:
        show(f"future {h} up", fit_score(*window(tr, h, True), *window(va, h, True)))
    for h in UPDS:
        show(f"past {h} up", fit_score(*window(tr, -h, True), *window(va, -h, True)))
