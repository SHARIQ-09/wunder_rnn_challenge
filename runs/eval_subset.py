"""Evaluate a checkpoint on the same seeded random valid subset train.py uses."""
import sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import StreamingSequenceDataset
from src.evaluate import load_pytorch_model
from src.train import evaluate_model
ckpt, n, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
ds = StreamingSequenceDataset("datasets/valid.parquet")
idx = sorted(np.random.default_rng(seed).choice(len(ds), size=n, replace=False).tolist())
m = load_pytorch_model(Path(ckpt), torch.device("cuda"))
r = evaluate_model(m, ds, torch.device("cuda"), chunk_size=2048, indices=idx)
print(f"{ckpt} subset{n}/seed{seed} WP={r['weighted_pearson']:.5f} t0={r['t0']:.5f} t1={r['t1']:.5f}")
