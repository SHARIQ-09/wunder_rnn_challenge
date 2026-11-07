"""Step 2: are the 22 price-like columns views of a low-rank discrete state? Within sequence, on price-update rows, per target-unit subgroup."""
import sys
from pathlib import Path
import numpy as np, pyarrow.parquet as pq
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils import FEATURE_COLUMNS as F
UNITS = (0.6569, 0.2413, 0.46855, 0.12543); N = 30


def rank(D, tot):
    """Number of singular directions needed for tot of the energy."""
    if len(D) < 30:
        return -1
    s = np.linalg.svd(D - D.mean(0), compute_uv=False) ** 2
    return int(np.searchsorted(np.cumsum(s) / s.sum(), tot) + 1)


def col_step(x):
    """Smallest non-trivial |delta| of a column and the share of deltas that are integer multiples of it."""
    a = np.abs(x[np.abs(x) > 1e-6])
    if len(a) < 10:
        return np.nan, np.nan
    u = np.unique(np.round(a, 5)); st = u[0]
    return st, float((np.abs(a / st - np.round(a / st)) < 0.02).mean())


for split in ("train", "valid"):
    pf = pq.ParquetFile(f"datasets/{split}.parquet"); lab = np.load(f"runs/decode/{split}_unit.npy")
    rng = np.random.default_rng(0)
    print(f"\n===== {split} =====")
    for k, u in enumerate(UNITS):
        ids = np.flatnonzero(lab == k)
        if not len(ids):
            continue
        rows = []
        for rg in rng.choice(ids, min(N, len(ids)), replace=False):
            d = pf.read_row_group(int(rg), columns=F).to_pandas().to_numpy(np.float64)
            for lo, nm in ((0, "i0"), (52, "i1")):
                p = d[:, lo:lo + 22]; dp = np.diff(p, axis=0); up = np.any(dp != 0, axis=1); D = dp[up]
                if len(D) < 30:
                    continue
                vec, c = np.unique(np.round(D, 4), axis=0, return_counts=True)
                st = [col_step(D[:, j]) for j in range(22)]
                # Moves on the two sides together, and share of updates touching only one side
                b, a = np.any(D[:, :11] != 0, 1), np.any(D[:, 11:] != 0, 1)
                rows.append(dict(i=nm, upd=up.mean(), r95=rank(D, .95), r99=rank(D, .99), r999=rank(D, .999), rb=rank(D[:, :11], .99), ra=rank(D[:, 11:], .99),
                                 distinct=len(vec) / len(D), top5=np.sort(c)[-5:].sum() / len(D), step=np.nanmedian([s[0] for s in st]), lat=np.nanmedian([s[1] for s in st]),
                                 both=(b & a).mean(), bonly=(b & ~a).mean(), aonly=(a & ~b).mean(), nunique=np.median([len(np.unique(p[:, j])) for j in range(22)])))
        for nm in ("i0", "i1"):
            R = [r for r in rows if r["i"] == nm]
            if R:
                med = {x: float(np.median([r[x] for r in R])) for x in R[0] if x != "i"}
                print(f"unit {u:.4f} {nm} ({len(R)} seqs): " + " ".join(f"{x}={v:.3g}" for x, v in med.items()), flush=True)
