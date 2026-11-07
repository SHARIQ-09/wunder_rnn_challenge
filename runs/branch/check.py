"""BranchGRU sanity: branch inputs are the intended columns, chunked/step-by-step match a full run, branches are isolated."""
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.model import BranchGRU, ScaledWunderGRU

torch.manual_seed(0)
m = BranchGRU().eval()
for k, n in enumerate(m.names):
    print(n, "inputs", len(getattr(m, f"idx_{k}")))
allidx = torch.cat([getattr(m, f"idx_{k}") for k in range(4)]).tolist()
print("covers every input once:", sorted(allidx) == list(range(456)))
x = torch.randn(1, 500, 456)
with torch.no_grad():
    p, h = m(x)
    hh, a = None, []
    for t in range(500):
        q, hh = m(x[:, t:t + 1], hh)
        a.append(q)
    hc, c = None, []
    for s in range(0, 500, 128):
        q, hc = m(x[:, s:s + 128], hc)
        c.append(q)
print("step vs full", (torch.cat(a, 1) - p).abs().max().item(), "chunk vs full", (torch.cat(c, 1) - p).abs().max().item(), "state", tuple(h.shape))
ref = ScaledWunderGRU(input_dim=456, hidden_dim=128, use_moe=False)
gru = lambda M: sum(q.numel() for k, q in M.named_parameters() if "gru" in k)
print(f"params {sum(q.numel() for q in m.parameters()):,} vs B-style H128 {sum(q.numel() for q in ref.parameters()):,}; recurrent {gru(m):,} vs {gru(ref):,}")
x2 = x.clone()
x2[..., 22:44] += 1.0
with torch.no_grad():
    z1 = m.branch_proj[0](x[..., m.idx_0])
    z2 = m.branch_proj[0](x2[..., m.idx_0])
print("price branch unaffected by volume columns:", (z1 - z2).abs().max().item())
