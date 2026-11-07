"""Feasibility: features computed inside the ONNX graph (state carried as a tensor) vs current Python features.
Checks parity of the feature block and times the full per-step path for both. Random GRU weights."""
import sys, io, time, importlib.util, numpy as np, torch, torch.nn as nn, onnxruntime as ort, psutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runs.fast_features import W as W_np, SQ as SQ_np, A as A_np, M as M_np
from src.model import ScaledWunderGRU

class GraphFeatures(nn.Module):
    """state = [flag, prev_mid0, prev_mid1, E(13)] (16). flag=0 on the first step of a sequence."""
    def __init__(self):
        super().__init__()
        for n, v in [('W', W_np), ('SQ', SQ_np), ('A', A_np), ('M', M_np)]:
            self.register_buffer(n, torch.tensor(v, dtype=torch.float32))
        sel = torch.zeros(112); sel[0:22] = 1; self.register_buffer('sel0', sel)
        sel = torch.zeros(112); sel[52:74] = 1; self.register_buffer('sel1', sel)
    def forward(self, x, s):  # x (1,112), s (1,16)
        lin = x @ self.W; sq = (x * x) @ self.SQ
        mid0 = (lin[:, 0:1] + lin[:, 1:2]) * 0.5; mid1 = (lin[:, 2:3] + lin[:, 3:4]) * 0.5
        mf = (x[:, 44:48] * x[:, 48:52]).sum(1, keepdim=True)
        flag = s[:, 0:1]
        r0 = flag * (mid0 - s[:, 1:2]); r1 = flag * (mid1 - s[:, 2:3])
        o4, o5 = lin[:, 4:5], lin[:, 5:6]
        u = torch.cat([r0, r0, r0, o4, o4, o4, r1, mid0 - mid1, r0 * r0, lin[:, 7:8], mf, r0, o5], 1)
        E = s[:, 3:]
        E = flag * (E + self.A * (u - E)) + (1 - flag) * u
        z = torch.cat([lin[:, 1:2] - lin[:, 0:1], torch.sqrt(torch.clamp(sq - lin[:, 0:2] ** 2, min=0.0) + 1e-6),
                       o4, lin[:, 8:12], lin[:, 3:4] - lin[:, 2:3], mid0 - mid1, o4 - o5, lin[:, 6:8], mf, r0, E], 1)
        base = x - self.sel0 * mid0 - self.sel1 * mid1
        feats = torch.cat([base, z @ self.M], 1)
        new_s = torch.cat([torch.ones_like(flag), mid0, mid1, E], 1)
        return feats, new_s

class Full(nn.Module):
    def __init__(self, gru):
        super().__init__(); self.f = GraphFeatures(); self.g = gru
    def forward(self, x, s, h):
        feats, ns = self.f(x, s)
        p, nh = self.g(feats.unsqueeze(1), h)
        return p[:, 0], ns, nh

def sess_from(model, args, names, outs):
    buf = io.BytesIO(); torch.onnx.export(model, args, buf, input_names=names, output_names=outs, opset_version=17, do_constant_folding=True)
    o = ort.SessionOptions(); o.intra_op_num_threads = 1; o.inter_op_num_threads = 1
    o.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL; o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    o.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return ort.InferenceSession(buf.getvalue(), sess_options=o, providers=['CPUExecutionProvider'])

if __name__ == "__main__":
    import pyarrow.parquet as pq, utils
    psutil.Process().cpu_affinity([int(sys.argv[1]) if len(sys.argv) > 1 else 14])
    raw = pq.ParquetFile('datasets/valid.parquet').read_row_group(3, columns=utils.FEATURE_COLUMNS).to_pandas().to_numpy(np.float32)
    # parity of features block
    spec = importlib.util.spec_from_file_location('sol', 'submission/solution.py'); sol = importlib.util.module_from_spec(spec); spec.loader.exec_module(sol)
    pm = sol.PredictionModel(); pm._reset_features()
    fs = sess_from(GraphFeatures().eval(), (torch.zeros(1, 112), torch.zeros(1, 16)), ['x', 's'], ['f', 'ns'])
    s = np.zeros((1, 16), np.float32); mx = 0.0
    for t in range(20000):
        f, s = fs.run(None, {'x': raw[t:t + 1], 's': s})
        mx = max(mx, float(np.abs(f[0] - pm._extract_32_features(raw[t])).max()))
    print(f"graph-feature parity vs solution.py over 20000 steps: max diff {mx:.2e}")
    torch.manual_seed(0)
    results = {}
    for H in (128, 160):
        gru = ScaledWunderGRU(input_dim=144, hidden_dim=H, num_layers=2, use_moe=False, dropout=0.0).eval()
        full = sess_from(Full(gru).eval(), (torch.zeros(1, 112), torch.zeros(1, 16), torch.zeros(2, 1, H)), ['x', 's', 'h'], ['p', 'ns', 'nh'])
        netonly = sess_from(gru, (torch.zeros(1, 1, 144), torch.zeros(2, 1, H)), ['f', 'h'], ['p', 'nh'])
        results[H] = (full, netonly)
    tm = {k: [] for k in ['old H128 (py feats + ORT)', 'graph H128', 'graph H160']}
    for rnd in range(4):
        full, net = results[128]; h = np.zeros((2, 1, 128), np.float32); pm._reset_features(); t0 = time.perf_counter_ns()
        for t in range(10000):
            f = pm._extract_32_features(raw[t]); p, h = net.run(None, {'f': f.reshape(1, 1, 144), 'h': h})
        tm['old H128 (py feats + ORT)'].append((time.perf_counter_ns() - t0) / 1e4 / 1e3)
        for H in (128, 160):
            full = results[H][0]; h = np.zeros((2, 1, H), np.float32); s = np.zeros((1, 16), np.float32); t0 = time.perf_counter_ns()
            for t in range(10000):
                p, s, h = full.run(None, {'x': raw[t:t + 1], 's': s, 'h': h})
            tm[f'graph H{H}'].append((time.perf_counter_ns() - t0) / 1e4 / 1e3)
    for k, v in tm.items(): print(f"{k:28s} {np.median(v):6.1f} us/step")
