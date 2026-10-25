"""Training pipeline for ScaledWunderGRU on Alpha Connectome (Wunder RNN Challenge).

Features:
- Causal feature engineering (112 raw + 23 microstructure = 135 features).
- Scaled multi-layer GRU with inter-layer LayerNorm residuals (hidden_dim=384).
- Truncated BPTT with hidden state preservation across chunks.
- Hybrid loss (Weighted MSE + batch-level negative Weighted Pearson).
- Mixed precision training (fp16) via torch.amp.autocast.
- Evaluates official Weighted Pearson metric using GlobalAccumulator.
- Cosine learning rate scheduling with warm-up.
- Saves best_model.pt and latest_model.pt to checkpoints/.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Setup paths
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
from src.model import ScaledWunderGRU, HybridLoss, WunderGRU


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ScaledWunderGRU on Alpha Connectome")
    parser.add_argument(
        "--train_path",
        type=str,
        default=str(ROOT_DIR / "datasets" / "train.parquet"),
        help="Path to train.parquet",
    )
    parser.add_argument(
        "--valid_path",
        type=str,
        default=str(ROOT_DIR / "datasets" / "valid.parquet"),
        help="Path to valid.parquet",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=str(ROOT_DIR / "checkpoints"),
        help="Directory to save model checkpoints",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Sequence batch size")
    parser.add_argument("--chunk_size", type=int, default=2048, help="BPTT chunk size")
    parser.add_argument("--input_dim", type=int, default=TOTAL_FEATURES, help="Input feature dimension (144: 112 raw + 32 engineered)")
    parser.add_argument("--hidden_dim", type=int, default=128, help="GRU hidden state dimension (default: 128)")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of GRU layers")
    parser.add_argument("--dropout", type=float, default=0.05, help="Dropout probability")
    parser.add_argument("--lr", type=float, default=1e-3, help="Peak learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for AdamW")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Max gradient norm")
    parser.add_argument("--mse_weight", type=float, default=1.0, help="Weight for Weighted MSE loss")
    parser.add_argument("--pearson_weight", type=float, default=1.0, help="Weight for Negative Pearson loss")
    parser.add_argument("--use_moe", action="store_true", default=False, help="Use Volatility-Gated MoE Dual Experts")
    parser.add_argument(
        "--val_sequences",
        type=int,
        default=50,
        help="Number of validation sequences to evaluate per epoch (-1 for full set)",
    )
    parser.add_argument(
        "--max_train_sequences",
        type=int,
        default=None,
        help="Limit number of train sequences (useful for quick debug/verification runs)",
    )
    parser.add_argument(
        "--no_amp",
        action="store_true",
        help="Disable automatic mixed precision (fp16)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint .pt to resume training from",
    )
    parser.add_argument(
        "--reset_epoch",
        action="store_true",
        help="Reset epoch counter and optimizer to start a fresh fine-tuning schedule",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for data shuffling and initialization",
    )
    parser.add_argument(
        "--checkpoint_prefix",
        type=str,
        default="best_model.pt",
        help="Filename for best checkpoint (default: best_model.pt)",
    )
    parser.add_argument(
        "--submission_dir",
        type=str,
        default=str(ROOT_DIR / "submission"),
        help="Directory to place solution.py and exported model.onnx",
    )
    parser.add_argument(
        "--submission_zip",
        type=str,
        default=str(ROOT_DIR / "submission.zip"),
        help="Path for packaged submission zip archive",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computation device (cuda/cpu)",
    )
    return parser.parse_args()


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    val_dataset: StreamingSequenceDataset,
    device: torch.device,
    feature_extractor: Optional[nn.Module] = None,
    chunk_size: int = 2048,
    max_sequences: int = 50,
    use_amp: bool = True,
) -> Dict[str, float]:
    """Evaluates the model on validation sequences using the official competition metric."""
    model.eval()
    accumulator = GlobalAccumulator()
    num_eval = min(len(val_dataset), max_sequences) if max_sequences > 0 else len(val_dataset)

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    t0 = time.time()
    for idx, batch in enumerate(val_loader):
        if idx >= num_eval:
            break

        raw_features = batch["features"].to(device)  # (1, 20000, 112)
        targets = batch["targets"].cpu().numpy()[0]  # (20000, 2)
        mask = batch["need_prediction"].cpu().numpy()[0]  # (20000,)
        if "is_scored" in batch:
            mask = mask & batch["is_scored"].cpu().numpy()[0]

        # Extract features causally if enabled
        if feature_extractor is not None:
            features, _ = feature_extractor(raw_features)
        else:
            features = raw_features

        preds = np.full((SEQUENCE_LENGTH, 2), np.nan, dtype=np.float32)
        hidden = model.init_hidden(batch_size=1, device=device)

        # Forward pass in chunks maintaining recurrent state
        for start_step in range(0, SEQUENCE_LENGTH, chunk_size):
            end_step = min(start_step + chunk_size, SEQUENCE_LENGTH)
            chunk_x = features[:, start_step:end_step]

            with torch.amp.autocast("cuda", enabled=(use_amp and device.type == "cuda")):
                chunk_pred, hidden = model(chunk_x, hidden)

            preds[start_step:end_step] = chunk_pred[0].float().cpu().numpy()

        accumulator.add(targets, preds, mask)

    metric_result = accumulator.result()
    metric_result["eval_time_sec"] = time.time() - t0
    metric_result["sequences_evaluated"] = num_eval
    return metric_result


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    feature_extractor: Optional[nn.Module] = None,
    chunk_size: int = 2048,
    grad_clip: float = 1.0,
    use_amp: bool = True,
    epoch: int = 1,
) -> Dict[str, float]:
    """Executes one training epoch using Truncated BPTT across chunked sub-sequences."""
    model.train()
    total_loss = 0.0
    total_mse = 0.0
    total_pearson_loss = 0.0
    num_batches = len(train_loader)
    total_chunks_processed = 0

    t0 = time.time()
    for batch_idx, batch in enumerate(train_loader):
        raw_features = batch["features"].to(device)         # (B, 20000, 112)
        targets = batch["targets"].to(device)               # (B, 20000, 2)
        need_pred = batch["need_prediction"].to(device)     # (B, 20000)

        # Causal feature computation on GPU if enabled
        with torch.no_grad():
            if feature_extractor is not None:
                features, _ = feature_extractor(raw_features)
            else:
                features = raw_features

        batch_size = features.size(0)
        # Reset hidden state for new sequences in this batch
        hidden = model.init_hidden(batch_size=batch_size, device=device)

        batch_loss = 0.0
        n_chunks = int(np.ceil(SEQUENCE_LENGTH / chunk_size))

        # Sequential Truncated BPTT through time
        for chunk_i in range(n_chunks):
            start_step = chunk_i * chunk_size
            end_step = min(start_step + chunk_size, SEQUENCE_LENGTH)

            chunk_x = features[:, start_step:end_step]
            chunk_y = targets[:, start_step:end_step]
            chunk_m = need_pred[:, start_step:end_step]

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(use_amp and device.type == "cuda")):
                pred, next_hidden = model(chunk_x, hidden)
                loss_dict = criterion(pred, chunk_y, chunk_m)
                loss = loss_dict["loss"]

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()

            # Detach hidden state to truncate backpropagation through time
            hidden = next_hidden.detach()

            # Track metrics
            batch_loss += loss.item()
            total_mse += loss_dict.get("weighted_mse", loss).item()
            total_pearson_loss += loss_dict.get("pearson_loss", torch.tensor(0.0)).item()
            total_chunks_processed += 1

        total_loss += (batch_loss / n_chunks)

        if (batch_idx + 1) % 25 == 0 or (batch_idx + 1) == num_batches:
            avg_loss = total_loss / (batch_idx + 1)
            avg_mse = total_mse / max(total_chunks_processed, 1)
            avg_pearson = total_pearson_loss / max(total_chunks_processed, 1)
            elapsed = time.time() - t0
            print(
                f"  Epoch [{epoch}] Batch [{batch_idx+1}/{num_batches}] "
                f"Loss: {avg_loss:.4f} (MSE: {avg_mse:.4f}, PLoss: {avg_pearson:.4f}) "
                f"({(batch_idx+1)*batch_size / elapsed:.1f} seq/s)",
                flush=True,
            )

    return {
        "train_loss": total_loss / max(num_batches, 1),
        "train_mse": total_mse / max(total_chunks_processed, 1),
        "train_pearson_loss": total_pearson_loss / max(total_chunks_processed, 1),
        "epoch_time_sec": time.time() - t0,
    }


def main():
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    use_amp = not args.no_amp and device.type == "cuda"

    print("=" * 60, flush=True)
    print("[TRAINING] Wunder RNN Challenge -- Scaled Training Pipeline", flush=True)
    print("=" * 60, flush=True)
    print(f"Device         : {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})", flush=True)
    print(f"Mixed Precision: {use_amp} (fp16)", flush=True)
    print(f"Batch Size     : {args.batch_size} sequences in parallel", flush=True)
    print(f"Chunk Size     : {args.chunk_size} timesteps per BPTT unroll", flush=True)
    print(f"Input Dim      : {args.input_dim} raw features", flush=True)
    print(f"GRU Capacity   : {args.num_layers} layers x {args.hidden_dim} hidden units (Residuals + LayerNorm)", flush=True)
    print(f"Learning Rate  : {args.lr} (AdamW + Cosine Decay)", flush=True)
    print(f"Loss Function  : HybridLoss (MSE: {args.mse_weight}, Pearson: {args.pearson_weight})", flush=True)
    print(f"Save Path      : {save_dir}", flush=True)
    print("=" * 60, flush=True)

    # 1. Datasets
    train_rg = list(range(args.max_train_sequences)) if args.max_train_sequences else None
    train_dataset = StreamingSequenceDataset(args.train_path, row_groups=train_rg)
    val_dataset = StreamingSequenceDataset(args.valid_path)

    print(f"Training Set   : {len(train_dataset):,} sequences available", flush=True)
    print(f"Validation Set : {len(val_dataset):,} sequences available", flush=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Safe on Windows
        pin_memory=(device.type == "cuda"),
    )

    # 2. Modules
    feature_extractor = None

    model = ScaledWunderGRU(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        output_dim=len(TARGET_COLUMNS),
        dropout=args.dropout,
        use_moe=args.use_moe,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {total_params:,} (~{(total_params*4)/(1024*1024):.2f} MB f32)")
    print(f"MoE Gated Experts: {args.use_moe}")

    criterion = HybridLoss(
        mse_weight=args.mse_weight,
        pearson_weight=args.pearson_weight,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.05,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_wp = -1.0
    start_epoch = 1
    epoch_checkpoints = []

    model_config = {
        "model_type": "ScaledWunderGRU",
        "input_dim": args.input_dim,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "output_dim": len(TARGET_COLUMNS),
        "dropout": args.dropout,
        "use_moe": args.use_moe,
    }

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    best_filename = args.checkpoint_prefix
    latest_filename = f"latest_{args.checkpoint_prefix}"
    best_path = save_dir / best_filename
    latest_path = save_dir / latest_filename

    # Resume if requested
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint and not args.reset_epoch:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "val_score" in checkpoint and not args.reset_epoch:
            best_val_wp = checkpoint["val_score"]
        if "epoch" in checkpoint and not args.reset_epoch:
            start_epoch = checkpoint["epoch"] + 1
        else:
            start_epoch = 1
        prior_score = checkpoint.get("val_score", 0.0)
        print(f"[RESUME] Loaded checkpoint from {args.resume} (prior WP: {prior_score:.4f}, reset_epoch={args.reset_epoch})", flush=True)

    # 3. Training Loop
    print("\nStarting training loop...", flush=True)
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---", flush=True)

        # Train 1 epoch
        train_metrics = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            device=device,
            feature_extractor=feature_extractor,
            chunk_size=args.chunk_size,
            grad_clip=args.grad_clip,
            use_amp=use_amp,
            epoch=epoch,
        )

        scheduler.step()

        # Validation
        print(f"Evaluating validation metric (on {args.val_sequences} sequences)...", flush=True)
        val_metrics = evaluate_model(
            model=model,
            val_dataset=val_dataset,
            device=device,
            feature_extractor=feature_extractor,
            chunk_size=args.chunk_size,
            max_sequences=args.val_sequences,
            use_amp=use_amp,
        )

        val_wp = val_metrics["weighted_pearson"]
        val_t0 = val_metrics["t0"]
        val_t1 = val_metrics["t1"]

        print(
            f"Epoch {epoch} Results:\n"
            f"  Train Loss: {train_metrics['train_loss']:.4f} (MSE: {train_metrics['train_mse']:.4f}, PLoss: {train_metrics['train_pearson_loss']:.4f}, time: {train_metrics['epoch_time_sec']:.1f}s)\n"
            f"  Val WP    : {val_wp:.4f} (t0: {val_t0:.4f}, t1: {val_t1:.4f}, time: {val_metrics['eval_time_sec']:.1f}s)\n"
            f"  Current LR: {scheduler.get_last_lr()[0]:.6f}",
            flush=True,
        )

        # Track epoch weights for SWA Ensemble
        epoch_checkpoints.append({k: v.cpu().clone() for k, v in model.state_dict().items()})

        # Save latest checkpoint
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_score": val_wp,
                "model_config": model_config,
            },
            latest_path,
        )

        # Save best checkpoint
        if val_wp > best_val_wp:
            best_val_wp = val_wp
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_score": val_wp,
                    "model_config": model_config,
                },
                best_path,
            )
            print(f"  [BEST] New best validation score! Saved to: {best_path}", flush=True)

    # 4. SWA (Stochastic Weight Averaging) Ensemble across Epoch Checkpoints
    if len(epoch_checkpoints) >= 2:
        print("\n" + "=" * 60, flush=True)
        print("Computing Stochastic Weight Averaged (SWA) Ensemble...", flush=True)
        swa_state = {}
        for k in epoch_checkpoints[0].keys():
            swa_state[k] = torch.stack([ckpt[k].float() for ckpt in epoch_checkpoints]).mean(dim=0)

        # Load SWA weights into model and evaluate
        model.load_state_dict({k: v.to(device) for k, v in swa_state.items()})
        swa_metrics = evaluate_model(
            model=model,
            val_dataset=val_dataset,
            device=device,
            chunk_size=args.chunk_size,
            max_sequences=args.val_sequences,
            use_amp=use_amp,
        )
        swa_wp = swa_metrics["weighted_pearson"]
        swa_t0 = swa_metrics["t0"]
        swa_t1 = swa_metrics["t1"]
        print(f"SWA Ensemble Validation WP: {swa_wp:.4f} (t0: {swa_t0:.4f}, t1: {swa_t1:.4f}) vs Best Epoch: {best_val_wp:.4f}", flush=True)

        if swa_wp >= best_val_wp:
            best_val_wp = swa_wp
            torch.save(
                {
                    "epoch": "swa_ensemble",
                    "model_state_dict": swa_state,
                    "val_score": swa_wp,
                    "model_config": model_config,
                },
                best_path,
            )
            print(f"  [SWA ENSEMBLE SELECTED] SWA outperformed single checkpoint! Saved to: {best_path}", flush=True)
        else:
            print("  [INFO] Single best checkpoint retained.", flush=True)

    print("\n" + "=" * 60, flush=True)
    print(f"Training Complete! Final Best Validation WP: {best_val_wp:.4f}", flush=True)
    print(f"Best checkpoint saved at: {best_path}", flush=True)
    print("=" * 60, flush=True)

    # 5. Automatically export best model to ONNX and package submission.zip
    try:
        import shutil
        from src.export_onnx import export_onnx
        from src.package_submission import package_submission

        sub_dir = Path(args.submission_dir)
        sub_dir.mkdir(parents=True, exist_ok=True)
        sol_dest = sub_dir / "solution.py"
        sol_src = ROOT_DIR / "submission" / "solution.py"
        if sol_src != sol_dest and sol_src.exists():
            shutil.copy(sol_src, sol_dest)

        onnx_path = sub_dir / "model.onnx"
        print(f"\n[AUTO-EXPORT] Exporting best model to ONNX: {onnx_path}...", flush=True)
        export_onnx(str(best_path), str(onnx_path))
        print(f"\n[AUTO-PACKAGE] Packaging verified submission archive: {args.submission_zip}...", flush=True)
        package_submission(submission_dir=str(sub_dir), output_zip=str(args.submission_zip))
    except Exception as e:
        print(f"[WARNING] Auto-export/packaging encountered: {e}", flush=True)


if __name__ == "__main__":
    main()

