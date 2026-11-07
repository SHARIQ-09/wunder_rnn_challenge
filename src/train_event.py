"""Train the event-clock GRU (src/event_model.py). Full-sequence BPTT on the event clock, loss on every row.

Example (screen recipe, comparable to runs/feat_screen.sh):
  python src/train_event.py --train_rg_range 0 2500 --epochs 2 --save_dir runs/event/screen_s42
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import FEATURE_COLUMNS, GlobalAccumulator
from src.event_model import EventGRU, event_inputs
from src.model import HybridLoss


class EventDataset(Dataset):
    def __init__(self, path, row_groups, scored=False):
        self.path, self.row_groups, self.scored, self._pf = path, list(row_groups), scored, None

    def __len__(self):
        return len(self.row_groups)

    def __getitem__(self, k):
        if self._pf is None:
            self._pf = pq.ParquetFile(self.path)
        cols = FEATURE_COLUMNS + ["t0", "t1", "need_prediction"] + (["is_scored"] if self.scored else [])
        d = self._pf.read_row_group(self.row_groups[k], columns=cols).to_pandas()
        ev_x, row_x, ev_index = event_inputs(d[FEATURE_COLUMNS].to_numpy(np.float32))
        item = {"ev_x": torch.from_numpy(ev_x), "row_x": torch.from_numpy(row_x), "ev_index": torch.from_numpy(ev_index),
                "y": torch.from_numpy(d[["t0", "t1"]].to_numpy(np.float32).copy()), "need": torch.from_numpy(d.need_prediction.values.copy())}
        if self.scored:
            item["scored"] = torch.from_numpy(d.is_scored.values.copy())
        return item


def collate(items):
    E = max(it["ev_x"].shape[0] for it in items)
    ev = torch.zeros(len(items), E, items[0]["ev_x"].shape[1])
    for b, it in enumerate(items):
        ev[b, :it["ev_x"].shape[0]] = it["ev_x"]
    out = {"ev_x": ev}
    for k in items[0]:
        if k != "ev_x":
            out[k] = torch.stack([it[k] for it in items])
    return out


def norm_stats(path, row_groups):
    ev_s, row_s = [], []
    pf = pq.ParquetFile(path)
    for rg in row_groups:
        raw = pf.read_row_group(int(rg), columns=FEATURE_COLUMNS).to_pandas().to_numpy(np.float32)
        ev_x, row_x, _ = event_inputs(raw)
        ev_s.append(ev_x[::2].astype(np.float64)); row_s.append(row_x[::10].astype(np.float64))
    ev_s, row_s = np.vstack(ev_s), np.vstack(row_s)
    f = lambda a: (torch.tensor(a.mean(0), dtype=torch.float32), torch.tensor(np.maximum(a.std(0), 1e-6), dtype=torch.float32))
    return f(ev_s), f(row_s)


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    acc = GlobalAccumulator()
    for b in loader:
        p = model(b["ev_x"].to(dev), b["row_x"].to(dev), b["ev_index"].to(dev)).float().cpu().numpy()
        for i in range(p.shape[0]):
            acc.add(b["y"][i].numpy(), p[i], (b["need"][i] & b["scored"][i]).numpy())
    model.train()
    return acc.result()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_path", default="datasets/train.parquet")
    ap.add_argument("--valid_path", default="datasets/valid.parquet")
    ap.add_argument("--train_rg_range", type=int, nargs=2, default=None)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--mse_weight", type=float, default=1.0)
    ap.add_argument("--pearson_weight", type=float, default=1.0)
    ap.add_argument("--val_sequences", type=int, default=150)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_dir", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save = Path(args.save_dir); save.mkdir(parents=True, exist_ok=True)
    n_train = pq.ParquetFile(args.train_path).num_row_groups
    train_rg = list(range(*args.train_rg_range)) if args.train_rg_range else list(range(n_train))
    n_val = pq.ParquetFile(args.valid_path).num_row_groups
    val_rg = sorted(np.random.default_rng(0).choice(n_val, args.val_sequences, replace=False).tolist())

    model = EventGRU(hidden_dim=args.hidden_dim, num_layers=args.num_layers).to(dev)
    t = time.time()
    (em, es), (rm, rs) = norm_stats(args.train_path, np.random.default_rng(1).choice(train_rg, min(100, len(train_rg)), replace=False))
    model.ev_mean.copy_(em); model.ev_std.copy_(es); model.row_mean.copy_(rm); model.row_std.copy_(rs)
    print(f"norm stats {time.time() - t:.0f}s | params {sum(p.numel() for p in model.parameters()):,} | train seqs {len(train_rg)}", flush=True)

    train_loader = DataLoader(EventDataset(args.train_path, train_rg), batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate, num_workers=args.num_workers, persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(EventDataset(args.valid_path, val_rg, scored=True), batch_size=args.batch_size,
                            collate_fn=collate, num_workers=args.num_workers)
    crit = HybridLoss(mse_weight=args.mse_weight, pearson_weight=args.pearson_weight, aligned=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(train_loader))
    cfg = {"hidden_dim": args.hidden_dim, "num_layers": args.num_layers, "model": "EventGRU"}

    best = -1.0
    for ep in range(args.epochs):
        t, tot, n = time.time(), 0.0, 0
        for b in train_loader:
            pred = model(b["ev_x"].to(dev), b["row_x"].to(dev), b["ev_index"].to(dev))
            loss = crit(pred, b["y"].to(dev), b["need"].to(dev))["loss"]
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += loss.item(); n += 1
        r = evaluate(model, val_loader, dev)
        print(f"Epoch {ep + 1}/{args.epochs}  train loss {tot / n:.4f}  ({time.time() - t:.0f}s)", flush=True)
        print(f"  Val WP    : {r['weighted_pearson']:.4f} (t0: {r['t0']:.4f}, t1: {r['t1']:.4f})", flush=True)
        state = {"model_state_dict": model.state_dict(), "model_config": cfg, "epoch": ep + 1, "val_score": r["weighted_pearson"]}
        torch.save(state, save / "latest.pt")
        if r["weighted_pearson"] > best:
            best = r["weighted_pearson"]; torch.save(state, save / "best.pt")
    print(f"done, best val {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
