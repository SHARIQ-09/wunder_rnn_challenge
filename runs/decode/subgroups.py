"""Label every sequence by target unit, check purity and mix per split, score A+B per subgroup on the official mask."""
import sys
from pathlib import Path
import numpy as np, pyarrow.parquet as pq, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils import FEATURE_COLUMNS as F
from src.evaluate import load_pytorch_model
from src.feature_sets import build_features
UNITS = np.array([0.6569, 0.2413, 0.46855, 0.12543]); TOL = 2e-3
out = Path("runs/decode")


def wc(p, y):
    p = np.clip(p, -2, 2); y = np.clip(y, -2, 2); w = np.abs(y)
    a = p - np.average(p, weights=w); b = y - np.average(y, weights=w)
    return float((w * a * b).sum() / np.sqrt((w * a * a).sum() * (w * b * b).sum() + 1e-30))
wp = lambda P, Y: np.mean([wc(P[:, j], Y[:, j]) for j in range(2)])

for split in ("train", "valid"):
    pf = pq.ParquetFile(f"datasets/{split}.parquet"); G = np.load(f"runs/groups/{split}_group.npy")
    n = pf.num_row_groups; lab = np.zeros(n, int); pur = np.zeros(n); second = np.zeros(n)
    for rg in range(n):
        v = pf.read_row_group(rg, columns=["t0", "t1"]).to_pandas().to_numpy(np.float64).ravel(); v = v[v != 0]
        r = v[:, None] / UNITS
        hit = (np.abs(r - np.round(r)) < TOL * 5).mean(0)
        lab[rg] = hit.argmax(); pur[rg] = hit.max(); second[rg] = np.sort(hit)[-2]
    np.save(out / f"{split}_unit.npy", lab); np.save(out / f"{split}_purity.npy", pur)
    print(f"\n{split}: {n} seqs. purity min {pur.min():.4f} p1 {np.percentile(pur, 1):.4f} median {np.median(pur):.4f}; seqs with purity<0.99: {(pur < .99).sum()}")
    print("unit share: " + " ".join(f"{UNITS[k]:.4f}:{(lab == k).mean():.3f}" for k in range(4)))
    print("dv group x unit counts (rows=dv group):"); print(np.array([[((G == g) & (lab == k)).sum() for k in range(4)] for g in range(3)]))
    sys.stdout.flush()

dev = torch.device("cuda")
ms = [(load_pytorch_model(Path("checkpoints_norm_ft/best_norm_ft.pt"), dev), "v144"), (load_pytorch_model(Path("checkpoints_raw_ev_ema_s21/best.pt"), dev), "raw_ev_ema")]
pf = pq.ParquetFile("datasets/valid.parquet"); lab = np.load(out / "valid_unit.npy")
P, Y, L = [], [], []
with torch.no_grad():
    for rg in np.linspace(0, pf.num_row_groups - 1, 300).astype(int):
        d = pf.read_row_group(int(rg), columns=F + ["t0", "t1", "need_prediction", "is_scored"]).to_pandas(); mk = d.need_prediction.values & d.is_scored.values
        if not mk.any():
            continue
        raw = d[F].to_numpy(np.float32)
        p = np.mean([m(torch.from_numpy(build_features(raw, fs)).unsqueeze(0).to(dev))[0][0].float().cpu().numpy() for m, fs in ms], 0)
        P.append(p[mk]); Y.append(d[["t0", "t1"]].to_numpy(np.float64)[mk]); L.append(np.full(mk.sum(), lab[rg]))
P, Y, L = map(np.concatenate, (P, Y, L))
np.savez(out / "ab_valid300.npz", P=P, Y=Y, L=L)
print(f"\nA+B on 300 valid seqs, scored rows: all WP {wp(P, Y):.4f}")
wt = np.abs(np.clip(Y, -2, 2)).sum(1)
for k in range(4):
    s = L == k
    if s.sum() < 100:
        continue
    d, dt = P[s, 0] - P[s, 1], Y[s, 0] - Y[s, 1]; a, at = P[s].sum(1), Y[s].sum(1)
    print(f"unit {UNITS[k]:.4f}: rows {s.mean():.3f} metric weight {wt[s].sum() / wt.sum():.3f} WP {wp(P[s], Y[s]):.4f} corr diff {np.corrcoef(d, dt)[0, 1]:.4f} corr sum {np.corrcoef(a, at)[0, 1]:.4f} pred std {P[s].std():.3f} target std {Y[s].std():.3f}")
