"""Full-valid WP inside each target-unit population (labels runs/decode/valid_unit.npy) for equal-weight ensembles.
Usage: python runs/cls/eval_units.py NAME=CKPT[,CKPT...] ..."""
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils import FEATURE_COLUMNS, GlobalAccumulator
from src.evaluate import load_pytorch_model
from src.feature_sets import build_features

UNITS = {0: "0.6569", 1: "0.2413", 2: "0.4686", 3: "0.1254"}
dev = torch.device("cpu")
paths = sorted({c for a in sys.argv[1:] for c in a.split("=", 1)[1].split(",")})
gru = {c: (load_pytorch_model(Path(c), dev), torch.load(c, map_location="cpu").get("model_config", {}).get("feature_set", "v144")) for c in paths}
ens = {a.split("=", 1)[0]: a.split("=", 1)[1].split(",") for a in sys.argv[1:]}
unit = np.load("runs/decode/valid_unit.npy")
acc = {n: {u: GlobalAccumulator() for u in UNITS} for n in ens}
pf = pq.ParquetFile("datasets/valid.parquet")
with torch.no_grad():
    for rg in range(pf.num_row_groups):
        d = pf.read_row_group(rg, columns=FEATURE_COLUMNS + ["t0", "t1", "need_prediction", "is_scored"]).to_pandas()
        raw = d[FEATURE_COLUMNS].to_numpy(np.float32)
        y = d[["t0", "t1"]].to_numpy(np.float32); m = d.need_prediction.values & d.is_scored.values
        P = {c: mdl(torch.from_numpy(build_features(raw, fs)).unsqueeze(0).to(dev))[0][0].float().cpu().numpy() for c, (mdl, fs) in gru.items()}
        for n, cs in ens.items():
            acc[n][int(unit[rg])].add(y, np.mean([P[c] for c in cs], 0), m)
for n in ens:
    print(f"{n:10s} " + "  ".join(f"unit {UNITS[u]} {acc[n][u].result()['weighted_pearson']:.4f}" for u in UNITS), flush=True)
