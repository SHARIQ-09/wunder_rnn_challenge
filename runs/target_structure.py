"""Read-only: what are t0/t1? Correlate them with future/past mid-price changes at many horizons."""
import sys, numpy as np, pyarrow.parquet as pq
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import utils
pf = pq.ParquetFile('datasets/valid.parquet')
H = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]
res = {('t0', 'fut'): {h: [] for h in H}, ('t1', 'fut'): {h: [] for h in H}, ('t0', 'past'): {h: [] for h in H}, ('t1', 'past'): {h: [] for h in H}}
ac = {k: {l: [] for l in [1, 10, 100, 1000]} for k in ('t0', 't1')}
t01 = []
for rg in range(0, 1873, 37):
    df = pf.read_row_group(rg).to_pandas()
    x = df[utils.FEATURE_COLUMNS].to_numpy(np.float64)
    mid = (x[:, 0:11].mean(1) + x[:, 11:22].mean(1)) / 2
    t = {k: df[k].to_numpy(np.float64) for k in ('t0', 't1')}
    t01.append(np.corrcoef(t['t0'], t['t1'])[0, 1])
    for k in t:
        for h in H:
            fut = mid[h:] - mid[:-h]
            res[(k, 'fut')][h].append(np.corrcoef(t[k][:-h], fut)[0, 1])
            res[(k, 'past')][h].append(np.corrcoef(t[k][h:], fut)[0, 1])
        for l in ac[k]:
            ac[k][l].append(np.corrcoef(t[k][:-l], t[k][l:])[0, 1])
print(f"corr(t0,t1) mean over seqs = {np.nanmean(t01):.3f}")
for k in ('t0', 't1'):
    print(f"{k} autocorr:", {l: round(float(np.nanmean(v)), 3) for l, v in ac[k].items()})
    print(f"{k} vs FUTURE mid change (mid[t+h]-mid[t]):", {h: round(float(np.nanmean(v)), 3) for h, v in res[(k, 'fut')].items()})
    print(f"{k} vs PAST   mid change (mid[t]-mid[t-h]):  ", {h: round(float(np.nanmean(v)), 3) for h, v in res[(k, 'past')].items()})
