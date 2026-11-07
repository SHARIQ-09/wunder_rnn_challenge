"""Inside the container: unzip the submission to a fresh dir and run it like the scorer on the 3 reference sequences."""
import sys, zipfile, tempfile, importlib.util, time, numpy as np
from dataclasses import dataclass
@dataclass(frozen=True)
class DataPoint:
    seq_ix: int; step_in_seq: int; need_prediction: bool; state: np.ndarray
d = tempfile.mkdtemp(); zipfile.ZipFile(sys.argv[1]).extractall(d)
sys.path.insert(0, d)
spec = importlib.util.spec_from_file_location('solution', f'{d}/solution.py'); sol = importlib.util.module_from_spec(spec); spec.loader.exec_module(sol)
m = sol.PredictionModel(); ref = dict(np.load(sys.argv[2])); mx = 0.0; t = time.perf_counter()
for j in range(3):
    raw = ref[f'raw{j}']; seq = int(ref[f'seq{j}'])
    for s in range(len(raw)):
        p = m.predict(DataPoint(seq, s, s >= 99, raw[s]))
        if s < 99: assert p is None
        else: mx = max(mx, float(np.abs(p - ref[f'pred{j}'][s]).max()))
print(f"zip OK: max |pred - torch| = {mx:.2e}, {(time.perf_counter() - t) / 60000 * 1e6:.1f} us/step")
