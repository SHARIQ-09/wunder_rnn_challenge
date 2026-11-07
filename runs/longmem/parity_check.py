"""LongMemory parity: block-matrix scan vs naive loop, chunked vs full, step-by-step vs full; mem_dim=0 unchanged."""
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.model import LongMemory, ScaledWunderGRU
from src.evaluate import load_pytorch_model

torch.manual_seed(0)
scales = (20.0, 200.0, 2000.0, 10000.0)
mem = LongMemory(8, scales)
B, T, K = 2, 3000, 8
u = torch.randn(B, T, K, dtype=torch.float64).float()
state = torch.zeros(B, 128)
f, st = mem(u, state)

# naive loop (float64)
e = torch.zeros(B, 4, K, dtype=torch.float64); w = torch.zeros(B, 4, dtype=torch.float64)
a = torch.tensor([1 / s for s in scales], dtype=torch.float64)
ref = []
for t in range(T):
    e = (1 - a)[None, :, None] * e + a[None, :, None] * u[:, t, None, :].double()
    w = (1 - a)[None] * w + a[None]
    ref.append((e / w[..., None]).reshape(B, -1))
ref = torch.stack(ref, 1)
print("scan vs loop   max err", (f[..., :4 * K].double() - ref).abs().max().item())

# chunked with odd chunk sizes
outs, s = [], torch.zeros(B, 128)
for a0, a1 in [(0, 1), (1, 700), (700, 2048), (2048, 3000)]:
    o, s1 = mem(u[:, a0:a1], s); outs.append(o)
    s = torch.zeros(B, 128); s[:, :s1.size(1)] = s1
print("chunked vs full max err", (torch.cat(outs, 1) - f).abs().max().item())

# full model: chunked + step-by-step vs one-shot, on CPU
m = ScaledWunderGRU(input_dim=456, hidden_dim=128, mem_dim=24, use_moe=False, dropout=0.0).eval()
x = torch.randn(1, 600, 456)
with torch.no_grad():
    p_full, h_full = m(x)
    h = None; ps = []
    for t in range(600):
        p, h = m(x[:, t:t + 1], h); ps.append(p)
    p_step = torch.cat(ps, 1)
    h = None; pc = []
    for a0 in range(0, 600, 256):
        p, h = m(x[:, a0:a0 + 256], h); pc.append(p)
print("model step vs full max err", (p_step - p_full).abs().max().item(), "| chunk vs full", (torch.cat(pc, 1) - p_full).abs().max().item(),
      "| hidden slots", tuple(h_full.shape), "| params", sum(q.numel() for q in m.parameters()))

# mem_dim=0: an old checkpoint loads and gives finite output with 2 hidden slots
old = load_pytorch_model(Path("checkpoints_raw_ev_ema_s21/best.pt"), torch.device("cpu"))
with torch.no_grad():
    p, h = old(torch.randn(1, 50, 456))
print("old checkpoint ok:", tuple(h.shape), bool(torch.isfinite(p).all()))
