"""Score a checkpoint (stationary features) on FULL validation, official mask. Prints one line."""
import sys, numpy as np, torch, pyarrow.parquet as pq
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import GlobalAccumulator, FEATURE_COLUMNS, TARGET_COLUMNS
from src.feature_sets import build_features
from src.evaluate import load_pytorch_model

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
ckpt = sys.argv[1]
dev = torch.device("cuda")
m = load_pytorch_model(Path(ckpt), dev)
fset = torch.load(ckpt, map_location="cpu").get("model_config", {}).get("feature_set", "v144")
acc = GlobalAccumulator()
pf = pq.ParquetFile("datasets/valid.parquet")
cols = FEATURE_COLUMNS + list(TARGET_COLUMNS) + ["need_prediction", "is_scored"]
with torch.no_grad():
    for rg in range(pf.num_row_groups):
        df = pf.read_row_group(rg, columns=cols).to_pandas()
        raw = df[FEATURE_COLUMNS].to_numpy(np.float32)
        x = torch.from_numpy(build_features(raw, fset)).unsqueeze(0).to(dev)
        p = m(x)[0][0].float().cpu().numpy()
        acc.add(df[list(TARGET_COLUMNS)].to_numpy(np.float32), p, df.need_prediction.values & df.is_scored.values)
r = acc.result()
print(f"{ckpt} [{fset}]  FULL-VALID WP={r['weighted_pearson']:.6f} t0={r['t0']:.6f} t1={r['t1']:.6f} rows={r['selected_rows']}")
