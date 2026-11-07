"""Relative per-step cost of candidate architectures in ONNX Runtime (1 thread, pinned core).
Random weights; only timing matters. Runs are interleaved in rounds so background load hits all equally."""
import sys, time, io, numpy as np, torch, torch.nn as nn, onnxruntime as ort, psutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model import ScaledWunderGRU
psutil.Process().cpu_affinity([int(sys.argv[1]) if len(sys.argv) > 1 else 15])
torch.manual_seed(0)

class Wrap(nn.Module):
    def __init__(self, models):
        super().__init__(); self.models = nn.ModuleList(models)
    def forward(self, x, h):
        outs, hs, i = [], [], 0
        for m in self.models:
            L = m.num_layers
            p, nh = m(x, h[i:i + L]); outs.append(p); hs.append(nh); i += L
        return torch.stack(outs).mean(0), torch.cat(hs, 0)

def make(H, L, n=1, model_type='gru'):
    ms = [ScaledWunderGRU(input_dim=144, hidden_dim=H, num_layers=L, use_moe=False, dropout=0.0).eval() for _ in range(n)]
    w = Wrap(ms).eval()
    buf = io.BytesIO()
    torch.onnx.export(w, (torch.randn(1, 1, 144), torch.zeros(L * n, 1, H)), buf, input_names=['x', 'h'],
                      output_names=['p', 'nh'], opset_version=17, do_constant_folding=True)
    o = ort.SessionOptions(); o.intra_op_num_threads = 1; o.inter_op_num_threads = 1
    o.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL; o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    o.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return ort.InferenceSession(buf.getvalue(), sess_options=o, providers=['CPUExecutionProvider']), (L * n, 1, H)

variants = {
  'v5: H128 x2': (128, 2, 1), 'H160 x2': (160, 2, 1), 'H192 x2': (192, 2, 1), 'H256 x2': (256, 2, 1),
  'H128 x3': (128, 3, 1), 'H96 x2': (96, 2, 1), '2x ensemble H128 x2': (128, 2, 2), '2x ensemble H96 x2': (96, 2, 2),
}
sess = {k: make(*v) for k, v in variants.items()}
x = np.random.randn(1, 1, 144).astype(np.float32)
times = {k: [] for k in sess}
for rnd in range(6):
    for k, (s, hs) in sess.items():
        h = np.zeros(hs, np.float32); t = time.perf_counter_ns()
        for _ in range(3000):
            p, h = s.run(None, {'x': x, 'h': h})
        times[k].append((time.perf_counter_ns() - t) / 3000 / 1000)
base = np.median(times['v5: H128 x2'])
for k, v in times.items():
    print(f"{k:22s} ORT call median {np.median(v):6.1f} us   ratio vs v5 ORT {np.median(v) / base:4.2f}")
# feature extraction + python overhead in the real solution (fixed cost per step)
import importlib.util
spec = importlib.util.spec_from_file_location('sol', 'submission/solution.py'); sol = importlib.util.module_from_spec(spec); spec.loader.exec_module(sol)
pm = sol.PredictionModel(); st = np.random.rand(112).astype(np.float32)
t = time.perf_counter_ns()
for _ in range(20000): pm._extract_32_features(st)
print(f"feature extraction per step: {(time.perf_counter_ns() - t) / 20000 / 1000:.1f} us")
