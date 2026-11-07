"""Torch reference for a C-kernel ensemble: average of each checkpoint's predictions with its own feature set.
Usage: python runs/ckernel/make_reference_ens.py out.npz ckpt1.pt [ckpt2.pt ...]"""
import sys, numpy as np, torch, pyarrow.parquet as pq
from pathlib import Path
sys.path.insert(0, '.')
import utils
from src.feature_sets import build_features
from src.evaluate import load_pytorch_model
ckpts = sys.argv[2:]
ms = [load_pytorch_model(Path(c), torch.device('cpu')) for c in ckpts]
fsets = [torch.load(c, map_location='cpu').get('model_config', {}).get('feature_set', 'v144') for c in ckpts]
pf = pq.ParquetFile('datasets/valid.parquet'); out = {}
for j, rg in enumerate([0, 700, 1500]):
    df = pf.read_row_group(rg).to_pandas()
    raw = df[utils.FEATURE_COLUMNS].to_numpy(np.float32)
    with torch.no_grad():
        p = np.mean([m(torch.from_numpy(build_features(raw, f)).unsqueeze(0))[0][0].numpy() for m, f in zip(ms, fsets)], axis=0)
    out[f'raw{j}'] = raw; out[f'pred{j}'] = p; out[f'y{j}'] = df[['t0', 't1']].to_numpy(np.float32)
    out[f'mask{j}'] = (df.need_prediction.values & df.is_scored.values); out[f'seq{j}'] = int(df.seq_ix.iloc[0])
np.savez_compressed(sys.argv[1], **out); print('saved', sys.argv[1], 'feature sets', fsets)
