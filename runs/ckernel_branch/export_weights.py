"""Convert ScaledWunderGRU / BranchGRU checkpoints into weights.npz for this kernel.
Usage: python runs/ckernel_branch/export_weights.py out.npz ckpt1.pt [ckpt2.pt ...]"""
import sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1])]
from solution import model_param_order, branch_param_order  # noqa: E402
from src.model import BranchGRU  # noqa: E402

FEATURE_SET_IDS = {"v144": 0, "raw_ev_ema": 1}
out, ckpts = sys.argv[1], sys.argv[2:]
arrays = {}
for k, path in enumerate(ckpts):
    ck = torch.load(path, map_location="cpu")
    sd, cfg = dict(ck["model_state_dict"]), ck.get("model_config", {})
    if cfg.get("use_moe"):
        raise SystemExit(f"{path}: MoE head not supported")
    if float(sd.get("norm_clip", torch.tensor(0.0))) != 0.0:
        raise SystemExit(f"{path}: norm_clip not supported")
    fset, nf = cfg.get("feature_set", "v144"), int(cfg.get("input_dim", 144))
    H, L = int(cfg.get("hidden_dim", 128)), int(cfg.get("num_layers", 2))
    sd.setdefault("feat_mean", torch.zeros(nf))
    sd.setdefault("feat_std", torch.ones(nf))
    cls_head = cfg.get("model_type") == "BranchClsGRU"
    branch = cls_head or cfg.get("model_type") == "BranchGRU"
    for name in (branch_param_order("", mlp=cls_head) if branch else model_param_order("", L)):
        arrays[f"m{k}_{name}"] = sd[name].float().numpy()
    if cls_head:
        arrays[f"m{k}_cls_values"] = sd["cls_values"].float().numpy()
    if branch:
        ref = BranchGRU(input_dim=nf, hidden_dim=H)
        for b in range(4):
            arrays[f"m{k}_idx{b}"] = getattr(ref, f"idx_{b}").numpy().astype(np.int32)
    arrays[f"m{k}_H"], arrays[f"m{k}_L"] = np.array(H), np.array(L)
    arrays[f"m{k}_F"], arrays[f"m{k}_T"] = np.array(FEATURE_SET_IDS[fset]), np.array(2 if cls_head else int(branch))
    print(f"m{k}: {path} H={H} features={fset} branch={branch}")
np.savez(out, **arrays)
print("wrote", out)
