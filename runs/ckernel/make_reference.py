import sys, numpy as np, torch, pyarrow.parquet as pq
from pathlib import Path
sys.path.insert(0, '.')
import utils
from src.features import extract_32_features_sequence
from src.evaluate import load_pytorch_model
m = load_pytorch_model(Path(sys.argv[1]), torch.device('cpu'))
pf = pq.ParquetFile('datasets/valid.parquet'); out = {}
for j, rg in enumerate([0, 700, 1500]):
    df = pf.read_row_group(rg).to_pandas()
    raw = df[utils.FEATURE_COLUMNS].to_numpy(np.float32)
    with torch.no_grad(): p = m(torch.from_numpy(extract_32_features_sequence(raw)).unsqueeze(0))[0][0].numpy()
    out[f'raw{j}'] = raw; out[f'pred{j}'] = p; out[f'y{j}'] = df[['t0', 't1']].to_numpy(np.float32)
    out[f'mask{j}'] = (df.need_prediction.values & df.is_scored.values); out[f'seq{j}'] = int(df.seq_ix.iloc[0])
np.savez_compressed(sys.argv[2], **out); print('saved', sys.argv[2])
