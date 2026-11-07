"""Full-valid check of the event GRU as a third ensemble member.

A = v144 norm_ft, B = raw_ev_ema (platform ensemble, full training).
E = event GRU (screen recipe), C = normal GRU with the same screen recipe (control third member).
Usage: python runs/event/eval_event_ens.py EVENT_CKPT [CONTROL_CKPT]
"""
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils import FEATURE_COLUMNS, GlobalAccumulator
from src.evaluate import load_pytorch_model
from src.feature_sets import build_features
from src.event_model import EventGRU, event_inputs

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ev_ckpt = sys.argv[1]
ctrl = sys.argv[2] if len(sys.argv) > 2 else "runs/screen/fs_slim/best.pt"
gru = {}
for name, c in [("A", "checkpoints_norm_ft/best_norm_ft.pt"), ("B", "checkpoints_raw_ev_ema_s21/best.pt"), ("C", ctrl)]:
    gru[name] = (load_pytorch_model(Path(c), dev), torch.load(c, map_location="cpu").get("model_config", {}).get("feature_set", "v144"))
ck = torch.load(ev_ckpt, map_location="cpu")
ev_model = EventGRU(hidden_dim=ck["model_config"]["hidden_dim"], num_layers=ck["model_config"]["num_layers"]).to(dev).eval()
ev_model.load_state_dict(ck["model_state_dict"])

names = ["A", "B", "C", "E"]
combos = [(n,) for n in names] + [("A", "B"), ("A", "B", "E"), ("A", "B", "C"), ("A", "B", "C", "E"), ("C", "E")]
acc = {c: GlobalAccumulator() for c in combos}
pf = pq.ParquetFile("datasets/valid.parquet")
ev_frac = []
with torch.no_grad():
    for rg in range(pf.num_row_groups):
        d = pf.read_row_group(rg, columns=FEATURE_COLUMNS + ["t0", "t1", "need_prediction", "is_scored"]).to_pandas()
        raw = d[FEATURE_COLUMNS].to_numpy(np.float32)
        y = d[["t0", "t1"]].to_numpy(np.float32); m = d.need_prediction.values & d.is_scored.values
        P = {}
        for n, (mdl, fs) in gru.items():
            P[n] = mdl(torch.from_numpy(build_features(raw, fs)).unsqueeze(0).to(dev))[0][0].float().cpu().numpy()
        ev_x, row_x, idx = event_inputs(raw)
        ev_frac.append(ev_x.shape[0] / len(raw))
        P["E"] = ev_model(torch.from_numpy(ev_x)[None].to(dev), torch.from_numpy(row_x)[None].to(dev),
                          torch.from_numpy(idx)[None].to(dev))[0].float().cpu().numpy()
        for c in combos:
            acc[c].add(y, np.mean([P[n] for n in c], 0), m)
print(f"event GRU: {ev_ckpt} | control C: {ctrl} | events per row {np.mean(ev_frac):.3f}")
for c in combos:
    r = acc[c].result()
    print(f"{'+'.join(c):10s} WP={r['weighted_pearson']:.6f}  t0={r['t0']:.6f}  t1={r['t1']:.6f}")
