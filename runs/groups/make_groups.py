"""Label every train and valid sequence by its i0 dv resting value; save row-group lists per group."""
import sys
from pathlib import Path
import numpy as np, pyarrow.parquet as pq
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils import FEATURE_COLUMNS
GROUPS = (0.0253, 0.038, 0.076)
DV = FEATURE_COLUMNS[48:52]
out = Path("runs/groups")
for split in ("train", "valid"):
    pf = pq.ParquetFile(f"datasets/{split}.parquet")
    g = np.zeros(pf.num_row_groups, dtype=int); dist = np.zeros(pf.num_row_groups)
    for rg in range(pf.num_row_groups):
        v = pf.read_row_group(rg, columns=DV).to_pandas().to_numpy(); eq = (v == v[:, :1]).all(1)
        r = float(np.median(v[eq, 0])); d = [abs(r - x) for x in GROUPS]
        g[rg] = int(np.argmin(d)); dist[rg] = min(d)
    print(split, "seqs", len(g), "counts", np.bincount(g, minlength=3), "max distance from group value", dist.max(), flush=True)
    for k in range(3):
        np.save(out / f"{split}_g{k}.npy", np.flatnonzero(g == k))
    np.save(out / f"{split}_group.npy", g)
