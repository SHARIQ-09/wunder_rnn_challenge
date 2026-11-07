"""Prototype: vectorized 144-feature extractor (stationary version), parity + speed vs submission/solution.py."""
import sys, time, importlib.util, numpy as np, psutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

W = np.zeros((112, 16), dtype=np.float64)
W[0:11, 0] = 1 / 11; W[11:22, 1] = 1 / 11; W[52:63, 2] = 1 / 11; W[63:74, 3] = 1 / 11
W[22:33, 4] = 1; W[33:44, 4] = -1; W[74:85, 5] = 1; W[85:96, 5] = -1
W[44:48, 6] = 1; W[48:52, 7] = 1
W[22, 8] = 1; W[33, 8] = -1; W[24, 9] = 1; W[35, 9] = -1; W[26, 10] = 1; W[37, 10] = -1; W[32, 11] = 1; W[43, 11] = -1
SQ = np.zeros((112, 2)); SQ[0:11, 0] = 1 / 11; SQ[11:22, 1] = 1 / 11
A = np.array([0.2, 0.05, 0.01, 0.2, 0.05, 0.01, 0.05, 0.05, 0.05, 0.05, 0.05, 0.002, 0.05])
# u = [ret0,ret0,ret0,ofi0,ofi0,ofi0,ret1,cross,ret0^2,dv,mf,ret0,ofi1]
# engineered block layout from z = [spread0, sdb, sda, ofi0, l8,l9,l10,l11, spread1, cross, ofi0-ofi1, dp, dv, mf, ret0, E(13)]  (28)
M = np.zeros((28, 32))
for j, src in enumerate([0, 1, 2, 0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]): M[src, j] = 1
E0 = 15
for j, e in enumerate([0, 1, 2]): M[E0 + e, 16 + j] = 1
M[E0 + 0, 19] = 1; M[E0 + 1, 19] = -1
for j, e in enumerate([3, 4, 5]): M[E0 + e, 20 + j] = 1
M[E0 + 3, 23] = 1; M[E0 + 5, 23] = -1
M[E0 + 6, 24] = 1; M[E0 + 6, 25] = 1; M[E0 + 1, 25] = -1
for j, e in enumerate([7, 8, 9, 10, 11]): M[E0 + e, 26 + j] = 1
M[E0 + 12, 31] = 1; M[E0 + 4, 31] = -1

class FastFeatures:
    def __init__(self):
        self.E = None; self.prev = None; self.z = np.zeros(28); self.out = np.zeros(144, np.float32)
    def reset(self):
        self.E = None; self.prev = None
    def __call__(self, x32):
        x = x32.astype(np.float64)
        lin = x @ W; sq = (x * x) @ SQ
        mid0 = (lin[0] + lin[1]) * 0.5; mid1 = (lin[2] + lin[3]) * 0.5
        mf = x[44:48] @ x[48:52]
        if self.prev is None:
            r0 = r1 = 0.0
        else:
            r0 = mid0 - self.prev[0]; r1 = mid1 - self.prev[1]
        self.prev = (mid0, mid1)
        u = np.array([r0, r0, r0, lin[4], lin[4], lin[4], r1, mid0 - mid1, r0 * r0, lin[7], mf, r0, lin[5]])
        self.E = u if self.E is None else self.E + A * (u - self.E)
        z = self.z
        z[0] = lin[1] - lin[0]; z[1:3] = np.sqrt(np.maximum(sq - lin[0:2] ** 2, 0.0) + 1e-6)
        z[3] = lin[4]; z[4:8] = lin[8:12]; z[8] = lin[3] - lin[2]; z[9] = mid0 - mid1; z[10] = lin[4] - lin[5]
        z[11] = lin[6]; z[12] = lin[7]; z[13] = mf; z[14] = r0; z[15:] = self.E
        o = self.out
        o[:112] = x32; o[0:22] -= np.float32(mid0); o[52:74] -= np.float32(mid1)
        o[112:] = z @ M
        return o

if __name__ == "__main__":
    import pyarrow.parquet as pq, utils
    psutil.Process().cpu_affinity([int(sys.argv[1]) if len(sys.argv) > 1 else 14])
    spec = importlib.util.spec_from_file_location('sol', 'submission/solution.py'); sol = importlib.util.module_from_spec(spec); spec.loader.exec_module(sol)
    raw = pq.ParquetFile('datasets/valid.parquet').read_row_group(3, columns=utils.FEATURE_COLUMNS).to_pandas().to_numpy(np.float32)
    pm = sol.PredictionModel(); pm._reset_features(); ff = FastFeatures()
    mx = 0.0
    for t in range(20000):
        a = pm._extract_32_features(raw[t]).copy(); b = ff(raw[t]); mx = max(mx, float(np.abs(a - b).max()))
    print(f"parity over 20000 steps: max |old - new| = {mx:.2e}")
    for name, fn in [('old', lambda r: pm._extract_32_features(r)), ('new', ff)]:
        ts = []
        for rnd in range(3):
            t0 = time.perf_counter_ns()
            for t in range(20000): fn(raw[t])
            ts.append((time.perf_counter_ns() - t0) / 20000 / 1000)
        print(f"{name}: {np.median(ts):.1f} us/step")
