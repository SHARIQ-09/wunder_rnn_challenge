"""Standalone Validation Evaluator for Wunder RNN Challenge.

Evaluates trained ScaledWunderGRU checkpoints on valid.parquet using the official
Weighted Pearson (WP) metric and prints a benchmark comparison against the 0.617052 baseline.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils import (
    SEQUENCE_LENGTH,
    WARMUP,
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    GlobalAccumulator,
)
from src.dataset import StreamingSequenceDataset
from src.features import CausalFeatureExtractor, TOTAL_FEATURES
from src.model import ScaledWunderGRU

BASELINE_WP = 0.617052


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate ScaledWunderGRU checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(ROOT_DIR / "checkpoints" / "best_model.pt"),
        help="Path to trained PyTorch checkpoint (.pt)",
    )
    parser.add_argument(
        "--valid_path",
        type=str,
        default=str(ROOT_DIR / "datasets" / "valid.parquet"),
        help="Path to valid.parquet",
    )
    parser.add_argument(
        "--num_sequences",
        type=int,
        default=-1,
        help="Number of sequences to validate (-1 for all 1,873 sequences)",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=2048,
        help="BPTT chunk size for forward evaluation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Evaluation device (cuda/cpu)",
    )
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(args.device)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"[ERROR] Checkpoint not found at: {checkpoint_path}")
        sys.exit(1)

    print("=" * 65)
    print("  Wunder RNN Challenge -- Standalone Validation Evaluator")
    print("=" * 65)
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model_config = checkpoint.get("model_config", {})
    input_dim = model_config.get("input_dim", TOTAL_FEATURES)
    hidden_dim = model_config.get("hidden_dim", 128)
    num_layers = model_config.get("num_layers", 2)
    output_dim = model_config.get("output_dim", len(TARGET_COLUMNS))
    use_moe = model_config.get("use_moe", True)
    saved_score = checkpoint.get("val_score", None)
    epoch = checkpoint.get("epoch", None)

    print(f"Model Architecture: ScaledWunderGRU (MoE: {use_moe})")
    print(f"  Input Dim       : {input_dim}")
    print(f"  Hidden Dim      : {hidden_dim}")
    print(f"  Layers          : {num_layers}")
    print(f"  Trained Epochs  : {epoch}")
    if saved_score is not None:
        print(f"  Checkpoint WP   : {saved_score:.4f}")
    print(f"Device            : {device}")
    print("=" * 65)

    # Initialize model
    model = ScaledWunderGRU(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_dim=output_dim,
        use_moe=use_moe,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Feature extractor if model expects 135 features
    feature_extractor = CausalFeatureExtractor().to(device) if input_dim > 112 else None

    # Load validation dataset
    val_dataset = StreamingSequenceDataset(args.valid_path, extract_features=(input_dim > 112))
    total_val_seqs = len(val_dataset)
    num_eval = total_val_seqs if args.num_sequences < 0 else min(args.num_sequences, total_val_seqs)

    print(f"Evaluating {num_eval:,} of {total_val_seqs:,} validation sequences...")

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    accumulator = GlobalAccumulator()
    t_start = time.time()

    for idx, batch in enumerate(val_loader):
        if idx >= num_eval:
            break

        raw_features = batch["features"].to(device)
        targets = batch["targets"].cpu().numpy()[0]  # (20000, 2)
        mask = batch["need_prediction"].cpu().numpy()[0]  # (20000,)
        if "is_scored" in batch:
            mask = mask & batch["is_scored"].cpu().numpy()[0]

        # Feature computation
        if feature_extractor is not None and raw_features.size(-1) == 112:
            features, _ = feature_extractor(raw_features)
        else:
            features = raw_features

        preds = np.full((SEQUENCE_LENGTH, 2), np.nan, dtype=np.float32)
        hidden = model.init_hidden(batch_size=1, device=device)

        # Forward pass in chunks
        for start_step in range(0, SEQUENCE_LENGTH, args.chunk_size):
            end_step = min(start_step + args.chunk_size, SEQUENCE_LENGTH)
            chunk_x = features[:, start_step:end_step]

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                chunk_pred, hidden = model(chunk_x, hidden)

            preds[start_step:end_step] = chunk_pred[0].float().cpu().numpy()

        accumulator.add(targets, preds, mask)

        if (idx + 1) % 100 == 0 or (idx + 1) == num_eval:
            elapsed = time.time() - t_start
            print(f"  Processed {idx + 1:,} / {num_eval:,} sequences ({(idx + 1) / elapsed:.1f} seq/s)")

    results = accumulator.result()
    total_time = time.time() - t_start

    wp_score = results["weighted_pearson"]
    t0_score = results["t0"]
    t1_score = results["t1"]
    delta_baseline = wp_score - BASELINE_WP

    print("\n" + "=" * 65)
    print("                    VALIDATION RESULTS")
    print("=" * 65)
    print(f"  Weighted Pearson (WP) : {wp_score:.6f}")
    print(f"    - Target 0 (t0)     : {t0_score:.6f}")
    print(f"    - Target 1 (t1)     : {t1_score:.6f}")
    print(f"  Baseline Reference    : {BASELINE_WP:.6f}")
    sign = "+" if delta_baseline >= 0 else ""
    print(f"  Delta vs Baseline     : {sign}{delta_baseline:.6f} WP")
    print(f"  Sequences Evaluated   : {num_eval:,}")
    print(f"  Total Wall Time       : {total_time:.1f}s ({num_eval / total_time:.1f} seq/s)")
    print("=" * 65)


if __name__ == "__main__":
    main()
