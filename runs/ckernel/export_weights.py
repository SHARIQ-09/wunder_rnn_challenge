"""Convert one or more ScaledWunderGRU checkpoints (v144 or raw_ev_ema features, no MoE) into weights.npz
for the C-kernel solution. Usage: python runs/ckernel/export_weights.py out.npz ckpt1.pt [ckpt2.pt ...]"""
import sys
import numpy as np
import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from solution import model_param_order  # noqa: E402

FEATURE_SET_IDS = {"v144": 0, "raw_ev_ema": 1}

out, ckpts = sys.argv[1], sys.argv[2:]
arrays = {}
for k, path in enumerate(ckpts):
    ck = torch.load(path, map_location="cpu")
    sd, cfg = ck["model_state_dict"], ck.get("model_config", {})
    if cfg.get("use_moe"):
        raise SystemExit(f"{path}: MoE head not supported")
    if float(sd.get("norm_clip", torch.tensor(0.0))) != 0.0:
        raise SystemExit(f"{path}: norm_clip not supported")
    L, H = int(cfg.get("num_layers", 2)), int(cfg.get("hidden_dim", 128))
    fset = cfg.get("feature_set", "v144")
    if fset not in FEATURE_SET_IDS:
        raise SystemExit(f"{path}: feature set {fset} not supported by the kernel")
    nf = int(cfg.get("input_dim", 144))
    sd = dict(sd)
    sd.setdefault("feat_mean", torch.zeros(nf))
    sd.setdefault("feat_std", torch.ones(nf))
    for name in model_param_order("", L):
        arrays[f"m{k}_{name}"] = sd[name].float().numpy()
    arrays[f"m{k}_H"] = np.array(H)
    arrays[f"m{k}_L"] = np.array(L)
    arrays[f"m{k}_F"] = np.array(FEATURE_SET_IDS[fset])
    print(f"m{k}: {path} H={H} L={L} features={fset}")
np.savez(out, **arrays)
print("wrote", out)
