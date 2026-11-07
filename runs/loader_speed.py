import sys, time, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import StreamingSequenceDataset
if __name__ == "__main__":
    ds = StreamingSequenceDataset('datasets/train.parquet', row_groups=list(range(5000, 5160)))
    for nw in (0, 4):
        dl = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=True, num_workers=nw, persistent_workers=nw > 0)
        t = time.time(); n = 0
        for b in dl: n += b['features'].shape[0]
        print(f"num_workers={nw}: {n / (time.time() - t):.1f} seq/s (includes worker startup)", flush=True)
