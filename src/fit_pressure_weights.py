"""Fit the frozen weights used by the "pressure" feature set (train data only).

Saves datasets/pressure_weights.npz:
- vmean, vstd (22): standardisation of the i0 volume columns (state 22..43).
- wB, wS (2, 11): |t|-weighted ridge weights from standardised bid / ask volumes to t0, t1
  (buy-pressure and sell-pressure scores, one per target).
- dstd (22), pcomp (22): scale of i0 price-column changes and their first principal component
  (a scalar "price move" per step, sign arbitrary).
- mv2_init: typical squared price move, used to start the running shock scale.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import FEATURE_COLUMNS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="datasets/train.parquet")
    ap.add_argument("--out", default="datasets/pressure_weights.npz")
    ap.add_argument("--sequences", type=int, default=300)
    ap.add_argument("--ridge", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pf = pq.ParquetFile(args.train)
    ids = np.random.default_rng(args.seed).choice(pf.num_row_groups, args.sequences, replace=False)
    V, Y, D = [], [], []
    for i in ids:
        df = pf.read_row_group(int(i), columns=FEATURE_COLUMNS + ["t0", "t1", "need_prediction"]).to_pandas()
        x = df[FEATURE_COLUMNS].to_numpy(np.float64)
        rows = np.flatnonzero(df.need_prediction.values)[::5]
        V.append(x[rows, 22:44]); Y.append(np.clip(df[["t0", "t1"]].to_numpy(np.float64)[rows], -2, 2))
        d = np.diff(x[:, :22], axis=0)
        D.append(d[np.any(d != 0, axis=1)])
    V, Y, D = np.vstack(V), np.vstack(Y), np.vstack(D)

    vmean, vstd = V.mean(0), V.std(0) + 1e-9
    Vz = (V - vmean) / vstd
    wB, wS = np.zeros((2, 11)), np.zeros((2, 11))
    for side, w_out in ((slice(0, 11), wB), (slice(11, 22), wS)):
        A = np.c_[Vz[:, side], np.ones(len(Vz))]
        for k in range(2):
            sw = np.abs(Y[:, k])
            G = (A * sw[:, None]).T @ A + args.ridge * np.diag(np.r_[np.ones(11), 0.0])
            w_out[k] = np.linalg.solve(G, (A * sw[:, None]).T @ Y[:, k])[:11]

    dstd = D.std(0) + 1e-12
    Dz = D / dstd
    pcomp = np.linalg.svd(Dz - Dz.mean(0), full_matrices=False)[2][0]
    mv2_init = float(np.mean((Dz @ pcomp) ** 2) * 0.2)  # mean over all steps; updates are ~20% of steps

    np.savez(args.out, vmean=vmean, vstd=vstd, wB=wB, wS=wS, dstd=dstd, pcomp=pcomp, mv2_init=mv2_init)
    print(f"rows {len(V)}  update steps {len(D)}  saved {args.out}")
    print("wB", np.round(wB, 3), "\nwS", np.round(wS, 3), "\nmv2_init", round(mv2_init, 4))


if __name__ == "__main__":
    main()
