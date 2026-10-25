"""Linear Weight Averaging (Model Soup) for Alpha Connectome models.

Merges weights of two or more linearly-connected models:
  W_ensemble = (W_1 + W_2 + ... + W_k) / k
Evaluates the averaged weights on the official competition metric,
exports to ONNX, and packages submission.zip. Zero latency overhead at inference time.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.dataset import StreamingSequenceDataset
from src.model import ScaledWunderGRU
from src.train import evaluate_model
from src.export_onnx import export_onnx
from src.package_submission import package_submission


def merge_checkpoints(
    checkpoint_paths: List[str],
    output_path: str,
    val_path: str = str(ROOT_DIR / "datasets" / "valid.parquet"),
    val_sequences: int = 100,
    weights: List[float] = None,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    device = torch.device(device_str)
    n = len(checkpoint_paths)
    if weights is None:
        weights = [1.0 / n] * n
    else:
        tot = sum(weights)
        weights = [w / tot for w in weights]

    print("=" * 60)
    print("      Model Soup (Linear Weight Averaging)")
    print("=" * 60)
    print(f"Blending {n} checkpoints with weights: {weights}")

    checkpoints = [torch.load(p, map_location="cpu") for p in checkpoint_paths]
    cfg = checkpoints[0]["model_config"]
    print(f"Model Configuration: {cfg}")

    # Build averaged state dict
    merged_state = {}
    param_keys = list(checkpoints[0]["model_state_dict"].keys())

    for k in param_keys:
        tensors = [ck["model_state_dict"][k].float() * w for ck, w in zip(checkpoints, weights)]
        merged_state[k] = sum(tensors)

    # Instantiate model and load merged weights
    model = ScaledWunderGRU(
        input_dim=cfg["input_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        output_dim=cfg["output_dim"],
        dropout=0.0,
        use_moe=cfg.get("use_moe", False),
    ).to(device)
    model.load_state_dict({k: v.to(device) for k, v in merged_state.items()})

    # Evaluate each parent model and merged model
    val_dataset = StreamingSequenceDataset(val_path)

    print("\nEvaluating Individual Parent Checkpoints:")
    for idx, (p, ck) in enumerate(zip(checkpoint_paths, checkpoints)):
        model.load_state_dict({k: v.to(device) for k, v in ck["model_state_dict"].items()})
        res = evaluate_model(model, val_dataset, device, max_sequences=val_sequences)
        print(f"  Parent [{idx+1}] {Path(p).name}: WP = {res['weighted_pearson']:.4f} (t0: {res['t0']:.4f}, t1: {res['t1']:.4f})")

    print("\nEvaluating Merged Model Soup:")
    model.load_state_dict({k: v.to(device) for k, v in merged_state.items()})
    soup_res = evaluate_model(model, val_dataset, device, max_sequences=val_sequences)
    soup_wp = soup_res["weighted_pearson"]
    print(f"  --> Merged Model Soup: WP = {soup_wp:.4f} (t0: {soup_res['t0']:.4f}, t1: {soup_res['t1']:.4f})")

    # Save merged checkpoint
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": "model_soup",
            "model_state_dict": merged_state,
            "val_score": soup_wp,
            "model_config": cfg,
        },
        out_p,
    )
    print(f"\nSaved merged checkpoint to: {out_p}")

    # Export to ONNX and package submission
    onnx_path = ROOT_DIR / "submission" / "model.onnx"
    print(f"\n[EXPORT] Exporting merged Model Soup to: {onnx_path}...")
    export_onnx(str(out_p), str(onnx_path))
    print("\n[PACKAGE] Building submission.zip...")
    package_submission()


def main():
    parser = argparse.ArgumentParser(description="Model Soup: Average weights of multiple models")
    parser.add_argument("--checkpoints", nargs="+", required=True, help="Paths to checkpoint .pt files")
    parser.add_argument("--output", type=str, default=str(ROOT_DIR / "checkpoints" / "best_model.pt"))
    parser.add_argument("--val_sequences", type=int, default=100)
    args = parser.parse_args()

    merge_checkpoints(args.checkpoints, args.output, val_sequences=args.val_sequences)


if __name__ == "__main__":
    main()
