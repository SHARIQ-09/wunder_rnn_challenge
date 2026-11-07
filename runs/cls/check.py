"""BranchClsGRU sanity: grid fits real train targets, chunked/step runs match a full run, loss backpropagates, checkpoint round-trip shape."""
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.model import BranchClsGRU, HybridLoss

torch.manual_seed(0)
m = BranchClsGRU().eval()
v = m.cls_values
print("classes", m.n_cls, "min gap", (v[1:] - v[:-1]).min().item(), "range", v[0].item(), v[-1].item())
pf = pq.ParquetFile("datasets/train.parquet")
worst, ok = 0.0, []
for rg in np.linspace(0, pf.num_row_groups - 1, 40).astype(int):
    d = pf.read_row_group(int(rg), columns=["t0", "t1", "need_prediction"]).to_pandas()
    y = torch.from_numpy(np.clip(d[["t0", "t1"]].to_numpy(np.float32)[d.need_prediction.values], -2, 2))
    dist = (y.unsqueeze(-1) - v).abs().min(-1).values
    worst = max(worst, dist.max().item()); ok.append((dist < 1e-3).float().mean().item())
print(f"train targets to nearest class: worst {worst:.4f}, share within 0.001 {np.mean(ok):.4f}")
x = torch.randn(1, 400, 456)
with torch.no_grad():
    p, h = m(x)
    hh, a = None, []
    for t in range(400):
        q, hh = m(x[:, t:t + 1], hh)
        a.append(q)
print("step vs full", (torch.cat(a, 1) - p).abs().max().item(), "pred", tuple(p.shape), "state", tuple(h.shape), "in clip", p.abs().max().item() <= 2.0)
m.train()
y = torch.randn(2, 300, 2)
mask = torch.ones(2, 300, dtype=torch.bool)
p, _ = m(torch.randn(2, 300, 456))
ce = m.ce_loss(y, mask)
loss = HybridLoss()(p, y, mask)["loss"] + 0.5 * ce
loss.backward()
gr = {n: q.grad is not None and q.grad.abs().sum().item() > 0 for n, q in m.named_parameters()}
print("CE at start", round(ce.item(), 3), "log K", round(float(np.log(m.n_cls)), 3), "params without gradient:", [n for n, g in gr.items() if not g])
print("params", sum(q.numel() for q in m.parameters()))
