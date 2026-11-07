"""Step 1: are t0, t1, sum, diff integer multiples of a unit, and is that unit fixed per group or per sequence?
Found on train sequences, rechecked on valid sequences with the units fitted on train."""
import sys
from pathlib import Path
import numpy as np, pyarrow.parquet as pq
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
N_SEQ, TOL = 200, 1e-3
NAMES = ("t0", "t1", "sum", "diff")


def load(split, n, offset=0):
    """Yield (group, targets (R, 4)) for n evenly spaced sequences, required rows only."""
    pf = pq.ParquetFile(f"datasets/{split}.parquet"); G = np.load(f"runs/groups/{split}_group.npy")
    for rg in np.linspace(offset, pf.num_row_groups - 1, n).astype(int):
        d = pf.read_row_group(int(rg), columns=["t0", "t1", "need_prediction"]).to_pandas()
        y = d[["t0", "t1"]].to_numpy(np.float64)[d.need_prediction.values]
        yield int(G[rg]), np.column_stack([y[:, 0], y[:, 1], y.sum(1), y[:, 0] - y[:, 1]])


def on_lattice(v, u):
    return np.abs(v / u - np.round(v / u)) < TOL


def best_unit(v):
    """Largest candidate unit that puts the most rows on a lattice. Candidates: distinct |values| and their small divisors."""
    vals, cnt = np.unique(np.round(np.abs(v[v != 0]), 4), return_counts=True)
    if not len(vals):
        return np.nan, 1.0
    cands = np.unique(np.round(np.concatenate([vals[np.argsort(-cnt)[:40]] / k for k in (1, 2, 3, 4, 5, 6, 8)]), 5))
    cands = cands[cands > 0.01]
    hit = np.array([(cnt * (np.abs(vals / c - np.round(vals / c)) < TOL)).sum() / cnt.sum() for c in cands])
    ok = np.flatnonzero(hit >= hit.max() - 0.005)
    return float(cands[ok[-1]]), float(hit[ok[-1]])


data = {"train": list(load("train", N_SEQ)), "valid": list(load("valid", 100, 1))}
print("== Pooled value tables, train ==")
units = {}
for k in range(3):
    seqs = [y for g, y in data["train"] if g == k]; Y = np.vstack(seqs)
    print(f"\ngroup {k}: {len(seqs)} seqs, {len(Y):,} rows")
    for j, nm in enumerate(NAMES):
        v = Y[:, j]; u, hit = best_unit(v); units[k, j] = u
        vals, cnt = np.unique(np.round(v, 4), return_counts=True); o = np.argsort(-cnt)[:8]
        top = " ".join(f"{vals[i]:+.4f}({cnt[i] / len(v):.3f})" for i in o)
        print(f"  {nm:4s} unit {u:.5f} on-lattice {hit:.4f} distinct {len(vals):4d} zero {(np.abs(v) < 1e-6).mean():.3f} | {top}")

print("\n== Per-sequence unit of diff and t0: is it fixed within a group? ==")
for k in range(3):
    for j in (0, 3):
        us = np.array([best_unit(y[:, j])[0] for g, y in data["train"] if g == k])
        vals, cnt = np.unique(np.round(us, 4), return_counts=True); o = np.argsort(-cnt)[:6]
        print(f"group {k} {NAMES[j]:4s}: " + " ".join(f"{vals[i]:.4f}x{cnt[i]}" for i in o) + f"  ({len(vals)} distinct)")

print("\n== Valid recheck with train units (share of rows on lattice, then integer histogram of diff) ==")
for k in range(3):
    Y = [y for g, y in data["valid"] if g == k]
    if not Y:
        continue
    Y = np.vstack(Y)
    print(f"group {k} ({len(Y):,} rows): " + " ".join(f"{NAMES[j]} {on_lattice(Y[:, j], units[k, j]).mean():.4f}" for j in range(4)))
    n = np.round(Y[:, 3] / units[k, 3]).astype(int); vals, cnt = np.unique(n, return_counts=True); o = np.argsort(-cnt)[:12]
    print("   diff integers: " + " ".join(f"{vals[i]}:{cnt[i] / len(n):.3f}" for i in o))
