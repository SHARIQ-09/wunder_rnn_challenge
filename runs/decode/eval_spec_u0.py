"""Unit-0.6569 specialist on full valid, official mask: within each population and globally, routed by the true label and by the causal warm-up detector.
Detector: dv group 0 (i0 dv resting value over rows 0-98, as runs/groups/make_groups.py) and at most 7 distinct i0 price rows in rows 0-98.
Usage: python runs/decode/eval_spec_u0.py SPECIALIST_CKPT"""
import sys
from pathlib import Path
import numpy as np, pyarrow.parquet as pq, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils import FEATURE_COLUMNS as F, GlobalAccumulator
from src.evaluate import load_pytorch_model
from src.feature_sets import build_features
dev = torch.device("cuda")
load = lambda c, fs: (load_pytorch_model(Path(c), dev), fs)
A, B, Br = load("checkpoints_norm_ft/best_norm_ft.pt", "v144"), load("checkpoints_raw_ev_ema_s21/best.pt", "raw_ev_ema"), load("checkpoints_branch80_s21/best.pt", "raw_ev_ema")
S = load(sys.argv[1], "raw_ev_ema")
GROUPS = np.array([0.0253, 0.038, 0.076])
UNITS = (0.6569, 0.2413, 0.46855, 0.12543)
lab = np.load("runs/decode/valid_unit.npy"); pf = pq.ParquetFile("datasets/valid.parquet")
names = ["B", "S", "A+B", "A+S", "A+Br", "A+Br+S"]
acc = {(n, k): GlobalAccumulator() for n in names for k in range(4)}
glob = {k: GlobalAccumulator() for k in ["B", "A+B", "A+Br", "B->S true", "A+B->A+S true", "A+Br->A+S true", "A+Br->A+S det", "A+Br->A+Br+S true", "A+Br->A+Br+S det"]}
tp = fp = fn = tn = 0
with torch.no_grad():
    for rg in range(pf.num_row_groups):
        d = pf.read_row_group(rg, columns=F + ["t0", "t1", "need_prediction", "is_scored"]).to_pandas()
        raw = d[F].to_numpy(np.float32); y = d[["t0", "t1"]].to_numpy(np.float32); m = d.need_prediction.values & d.is_scored.values
        f = lambda t: t[0](torch.from_numpy(build_features(raw, t[1])).unsqueeze(0).to(dev))[0][0].float().cpu().numpy()
        pa, pb, pr, ps = f(A), f(B), f(Br), f(S); k = int(lab[rg]); P = {"B": pb, "S": ps, "A+B": (pa + pb) / 2, "A+S": (pa + ps) / 2, "A+Br": (pa + pr) / 2, "A+Br+S": (pa + pr + ps) / 3}
        for n in names:
            acc[n, k].add(y, P[n], m)
        w = raw[:99]; dv = w[:, 48:52]; eq = (dv == dv[:, :1]).all(1)
        g = int(np.argmin(np.abs(np.median(dv[eq, 0]) - GROUPS))) if eq.any() else 0
        det = g == 0 and len(np.unique(w[:, :22], axis=0)) <= 7
        tp += det and k == 0; fp += det and k != 0; fn += (not det) and k == 0; tn += (not det) and k != 0
        tr = k == 0
        glob["B"].add(y, pb, m); glob["A+B"].add(y, P["A+B"], m); glob["A+Br"].add(y, P["A+Br"], m)
        glob["B->S true"].add(y, ps if tr else pb, m); glob["A+B->A+S true"].add(y, P["A+S"] if tr else P["A+B"], m)
        glob["A+Br->A+S true"].add(y, P["A+S"] if tr else P["A+Br"], m); glob["A+Br->A+S det"].add(y, P["A+S"] if det else P["A+Br"], m)
        glob["A+Br->A+Br+S true"].add(y, P["A+Br+S"] if tr else P["A+Br"], m); glob["A+Br->A+Br+S det"].add(y, P["A+Br+S"] if det else P["A+Br"], m)
for k, u in enumerate(UNITS):
    print(f"unit {u:.4f}: " + "  ".join(f"{n} {acc[n, k].result()['weighted_pearson']:.5f}" for n in names))
print(f"detector on valid: tp {tp} fp {fp} fn {fn} tn {tn}")
for n, v in glob.items():
    print(f"global {n}: {v.result()['weighted_pearson']:.6f}")
