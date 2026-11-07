"""Base vs specialist on full valid, by group and globally when routing that group to the specialist.
Usage: python runs/groups/eval_specialist.py GROUP SPECIALIST_CKPT [member A also averaged]"""
import sys
from pathlib import Path
import numpy as np, pyarrow.parquet as pq, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils import FEATURE_COLUMNS, GlobalAccumulator
from src.evaluate import load_pytorch_model
from src.feature_sets import build_features
G, ck = int(sys.argv[1]), sys.argv[2]
dev = torch.device("cuda")
load = lambda c, fs: (load_pytorch_model(Path(c), dev), fs)
A, B = load("checkpoints_norm_ft/best_norm_ft.pt", "v144"), load("checkpoints_raw_ev_ema_s21/best.pt", "raw_ev_ema")
S = load(ck, "raw_ev_ema")
groups = np.load("runs/groups/valid_group.npy")
pf = pq.ParquetFile("datasets/valid.parquet")
names = ["B", "S", "A+B", "A+S"]
acc = {(n, g): GlobalAccumulator() for n in names for g in range(3)}
glob = {k: GlobalAccumulator() for k in ["B", "B routed", "A+B", "A+S routed"]}
with torch.no_grad():
    for rg in range(pf.num_row_groups):
        d = pf.read_row_group(rg, columns=FEATURE_COLUMNS + ["t0", "t1", "need_prediction", "is_scored"]).to_pandas()
        raw = d[FEATURE_COLUMNS].to_numpy(np.float32); y = d[["t0", "t1"]].to_numpy(np.float32); m = d.need_prediction.values & d.is_scored.values
        f = lambda t: t[0](torch.from_numpy(build_features(raw, t[1])).unsqueeze(0).to(dev))[0][0].float().cpu().numpy()
        pa, pb, ps = f(A), f(B), f(S); g = groups[rg]
        P = {"B": pb, "S": ps, "A+B": (pa + pb) / 2, "A+S": (pa + ps) / 2}
        for n in names:
            acc[n, g].add(y, P[n], m)
        glob["B"].add(y, pb, m); glob["A+B"].add(y, P["A+B"], m)
        glob["B routed"].add(y, ps if g == G else pb, m); glob["A+S routed"].add(y, P["A+S"] if g == G else P["A+B"], m)
for g in range(3):
    print(f"group {g}: " + "  ".join(f"{n} {acc[n, g].result()['weighted_pearson']:.5f}" for n in names))
print("global: " + "  ".join(f"{k} {v.result()['weighted_pearson']:.6f}" for k, v in glob.items()))
