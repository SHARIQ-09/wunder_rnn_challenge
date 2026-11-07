"""Evaluate a checkpoint on valid row groups [start, end)."""
import sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import StreamingSequenceDataset
from src.evaluate import load_pytorch_model
from src.train import evaluate_model
ckpt, a, b = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
dev = torch.device("cuda")
m = load_pytorch_model(Path(ckpt), dev)
ds = StreamingSequenceDataset("datasets/valid.parquet", row_groups=list(range(a, b)))
r = evaluate_model(m, ds, dev, chunk_size=2048, max_sequences=-1)
print(f"{ckpt} valid[{a}:{b}] WP={r['weighted_pearson']:.5f} t0={r['t0']:.5f} t1={r['t1']:.5f}")
