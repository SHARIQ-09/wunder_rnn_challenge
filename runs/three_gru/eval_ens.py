"""Full-valid WP of every combination of GRU checkpoints (equal-weight average).

Usage: python runs/three_gru/eval_ens.py NAME=CKPT [NAME=CKPT ...]
"""
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils import FEATURE_COLUMNS, GlobalAccumulator
from src.evaluate import load_pytorch_model
from src.feature_sets import build_features

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gru = {}
for arg in sys.argv[1:]:
    name, c = arg.split("=", 1)
    gru[name] = (load_pytorch_model(Path(c), dev), torch.load(c, map_location="cpu").get("model_config", {}).get("feature_set", "v144"))
names = list(gru)
combos = [c for k in range(1, len(names) + 1) for c in combinations(names, k)]
acc = {c: GlobalAccumulator() for c in combos}
pf = pq.ParquetFile("datasets/valid.parquet")
with torch.no_grad():
    for rg in range(pf.num_row_groups):
        d = pf.read_row_group(rg, columns=FEATURE_COLUMNS + ["t0", "t1", "need_prediction", "is_scored"]).to_pandas()
        raw = d[FEATURE_COLUMNS].to_numpy(np.float32)
        y = d[["t0", "t1"]].to_numpy(np.float32); m = d.need_prediction.values & d.is_scored.values
        P = {n: mdl(torch.from_numpy(build_features(raw, fs)).unsqueeze(0).to(dev))[0][0].float().cpu().numpy()
             for n, (mdl, fs) in gru.items()}
        for c in combos:
            acc[c].add(y, np.mean([P[n] for n in c], 0), m)
for c in combos:
    r = acc[c].result()
    print(f"{'+'.join(c):12s} WP={r['weighted_pearson']:.6f}  t0={r['t0']:.6f}  t1={r['t1']:.6f}", flush=True)
