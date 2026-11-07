"""Score the average (each model with its own feature set) of several checkpoints (stationary features) on FULL validation; also every pair."""
import sys, itertools, numpy as np, torch, pyarrow.parquet as pq
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import GlobalAccumulator, FEATURE_COLUMNS, TARGET_COLUMNS
from src.feature_sets import build_features
from src.evaluate import load_pytorch_model

dev = torch.device("cuda")
ckpts = sys.argv[1:]
ms = [load_pytorch_model(Path(c), dev) for c in ckpts]
fsets = [torch.load(c, map_location="cpu").get("model_config", {}).get("feature_set", "v144") for c in ckpts]
subsets = [tuple(range(len(ms)))] + list(itertools.combinations(range(len(ms)), 2))
accs = {s: GlobalAccumulator() for s in subsets}
pf = pq.ParquetFile("datasets/valid.parquet")
cols = FEATURE_COLUMNS + list(TARGET_COLUMNS) + ["need_prediction", "is_scored"]
with torch.no_grad():
    for rg in range(pf.num_row_groups):
        df = pf.read_row_group(rg, columns=cols).to_pandas()
        raw = df[FEATURE_COLUMNS].to_numpy(np.float32)
        X = {f: torch.from_numpy(build_features(raw, f)).unsqueeze(0).to(dev) for f in set(fsets)}
        P = [m(X[f])[0][0].float().cpu().numpy() for m, f in zip(ms, fsets)]
        y = df[list(TARGET_COLUMNS)].to_numpy(np.float32); mk = df.need_prediction.values & df.is_scored.values
        for s in subsets:
            accs[s].add(y, np.mean([P[i] for i in s], axis=0), mk)
for s in subsets:
    print(f"{s}  WP={accs[s].result()['weighted_pearson']:.6f}")
