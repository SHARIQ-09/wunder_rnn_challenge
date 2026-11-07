"""Computes per-feature mean/std of the 144 engineered inputs over random train sequences.

Usage:
    python src/compute_norm_stats.py --num_sequences 1000 --out datasets/norm_stats.npz
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.dataset import StreamingSequenceDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default=str(ROOT_DIR / "datasets" / "train.parquet"))
    parser.add_argument("--num_sequences", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--feature_set", default="v144")
    parser.add_argument("--min_std", type=float, default=1e-6)
    parser.add_argument("--out", default=str(ROOT_DIR / "datasets" / "norm_stats.npz"))
    args = parser.parse_args()

    probe = StreamingSequenceDataset(args.train_path, include_targets=False)
    n_total = probe.total_row_groups
    rng = np.random.default_rng(args.seed)
    picks = np.sort(rng.choice(n_total, size=min(args.num_sequences, n_total), replace=False))
    ds = StreamingSequenceDataset(args.train_path, row_groups=picks.tolist(), include_targets=False,
                                  feature_set=args.feature_set)

    # Per-sequence moments merged with Chan's parallel algorithm (numerically stable).
    count = 0
    mean = None
    m2 = None
    t0 = time.time()
    for i in range(len(ds)):
        x = ds[i]["features"].numpy().astype(np.float64)
        n_b = x.shape[0]
        mean_b = x.mean(axis=0)
        m2_b = ((x - mean_b) ** 2).sum(axis=0)
        if mean is None:
            count, mean, m2 = n_b, mean_b, m2_b
        else:
            delta = mean_b - mean
            tot = count + n_b
            mean = mean + delta * (n_b / tot)
            m2 = m2 + m2_b + delta ** 2 * (count * n_b / tot)
            count = tot
        if (i + 1) % 100 == 0:
            print(f"[{i+1}/{len(ds)}] {time.time() - t0:.1f}s", flush=True)

    std = np.sqrt(m2 / max(count - 1, 1))
    std = np.maximum(std, args.min_std)
    np.savez(args.out, mean=mean.astype(np.float32), std=std.astype(np.float32),
             sequences=picks, rows=count)
    print(f"Saved {args.out}: rows={count:,}, std min={std.min():.3e} max={std.max():.3e}")


if __name__ == "__main__":
    main()
