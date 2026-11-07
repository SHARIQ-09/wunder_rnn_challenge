"""Official Evaluation and Parity Verification Suite for Wunder RNN Challenge.

Strictly adheres to:
1. utils.py GlobalAccumulator for all metric computations (zero custom clipping or manual weighting).
2. Rigorous Parity Checks (GPU runner vs full solution.py step-by-step through ScorerStepByStep, TF32 disabled).
3. Ground-truth platform anchoring against the official 0.617052 baseline.
4. Single-core pinned CPU latency profiling for the complete predict(data_point) inference pipeline.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import psutil
import pyarrow.parquet as pq
import torch
import torch.nn as nn

# Disable TF32 globally as required for exact numerical determinism
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils import (
    SEQUENCE_LENGTH,
    WARMUP,
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    DataPoint,
    GlobalAccumulator,
    ScorerStepByStep,
    validate_sequence,
)
from src.features import extract_32_features_sequence, TOTAL_FEATURES
from src.model import ScaledWunderGRU, BranchGRU, BranchClsGRU


def load_solution_module(solution_path: Path):
    """Dynamically loads a solution.py file given its path."""
    if not solution_path.exists():
        raise FileNotFoundError(f"solution.py not found at {solution_path}")
    spec = importlib.util.spec_from_file_location("solution_module", solution_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_pytorch_model(
    checkpoint_path: Path,
    device: torch.device,
) -> nn.Module:
    """Loads ScaledWunderGRU checkpoint."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    cfg = checkpoint.get("model_config", {})
    if cfg.get("model_type") in ("BranchGRU", "BranchClsGRU"):
        branch_cls = BranchClsGRU if cfg["model_type"] == "BranchClsGRU" else BranchGRU
        model = branch_cls(input_dim=cfg["input_dim"], hidden_dim=cfg["hidden_dim"], output_dim=cfg.get("output_dim", 2), dropout=0.0).to(device).eval()
        model.load_state_dict(checkpoint["model_state_dict"])
        return model
    model = ScaledWunderGRU(
        input_dim=cfg.get("input_dim", TOTAL_FEATURES),
        hidden_dim=cfg.get("hidden_dim", 128),
        num_layers=cfg.get("num_layers", 2),
        output_dim=cfg.get("output_dim", 2),
        dropout=0.0,
        use_moe=cfg.get("use_moe", True),
        mem_dim=cfg.get("mem_dim", 0),
        mem_scales=tuple(cfg.get("mem_scales", (20.0, 200.0, 2000.0, 10000.0))),
    ).to(device).eval()
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def run_latency_benchmark(solution_path: Path, valid_path: Path, steps: int = 20000, core_id: int = 0):
    """Benchmarks full predict() path pinned to 1 CPU core."""
    p = psutil.Process()
    p.cpu_affinity([core_id])
    print(f"\n[LATENCY BENCHMARK] Process pinned to CPU core: {p.cpu_affinity()}")

    sol_mod = load_solution_module(solution_path)
    model = sol_mod.PredictionModel()

    pf = pq.ParquetFile(str(valid_path))
    table = pf.read_row_group(0)
    seq, need = validate_sequence(table)
    feats = np.column_stack([table[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float32)

    # Warm-up 1000 steps
    for s in range(min(1000, len(feats))):
        dp = DataPoint(seq, s, bool(need[s]), feats[s])
        _ = model.predict(dp)

    # Benchmark full sequence
    model = sol_mod.PredictionModel()
    durations_ns = []
    bench_steps = min(steps, len(feats))
    for s in range(bench_steps):
        dp = DataPoint(seq, s, bool(need[s]), feats[s])
        t0 = time.perf_counter_ns()
        _ = model.predict(dp)
        t1 = time.perf_counter_ns()
        durations_ns.append(t1 - t0)

    durations_us = np.array(durations_ns, dtype=np.float64) / 1000.0
    mean_us = np.mean(durations_us)
    median_us = np.median(durations_us)
    p95_us = np.percentile(durations_us, 95)
    p99_us = np.percentile(durations_us, 99)
    total_min = (mean_us * 37_460_000) / (1e6 * 60)

    print(f"Results for: {solution_path.parent.name}")
    print(f"  Mean Latency       : {mean_us:.2f} us/step")
    print(f"  Median Latency     : {median_us:.2f} us/step")
    print(f"  P95 Latency        : {p95_us:.2f} us/step")
    print(f"  P99 Latency        : {p99_us:.2f} us/step")
    print(f"  Projected Test Time: {total_min:.2f} minutes (Budget: <= 60 min, <= 96 us/step)")
    return {"mean_us": mean_us, "median_us": median_us, "total_min": total_min}


def run_parity_check(
    checkpoint_path: Path,
    solution_path: Path,
    valid_path: Path,
    num_sequences: int = 5,
    device: str = "cuda",
):
    """Verifies numerical parity between GPU batch runner and CPU solution.py step-by-step."""
    dev = torch.device(device)
    print("\n" + "=" * 65)
    print("  PARITY CHECK: GPU Runner vs Step-by-Step solution.py")
    print("=" * 65)
    print(f"PyTorch CUDA TF32 matmul allowed: {torch.backends.cuda.matmul.allow_tf32}")
    print(f"PyTorch CuDNN TF32 allowed       : {torch.backends.cudnn.allow_tf32}")

    pt_model = load_pytorch_model(checkpoint_path, dev)
    sol_mod = load_solution_module(solution_path)
    sol_model = sol_mod.PredictionModel()

    pf = pq.ParquetFile(str(valid_path))
    acc_gpu = GlobalAccumulator()
    acc_step = GlobalAccumulator()
    max_step_diff = 0.0

    for seq_idx in range(num_sequences):
        table = pf.read_row_group(seq_idx)
        seq_id, need = validate_sequence(table)
        raw_feats = np.column_stack([table[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float32)
        targets = np.column_stack([table[c].to_numpy() for c in TARGET_COLUMNS]).astype(np.float32)
        scored = table["is_scored"].to_numpy().astype(bool) if "is_scored" in table.schema.names else np.ones(20000, bool)
        mask = need & scored

        # Step-by-step solution
        preds_step = np.full((20000, 2), np.nan, dtype=np.float32)
        for s in range(20000):
            dp = DataPoint(seq_id, s, bool(need[s]), raw_feats[s])
            val = sol_model.predict(dp)
            if need[s]:
                preds_step[s] = val

        # GPU runner
        feats_144 = extract_32_features_sequence(raw_feats) if pt_model.input_dim > 112 else raw_feats
        x_tensor = torch.from_numpy(feats_144).unsqueeze(0).to(dev)
        with torch.no_grad():
            preds_gpu_tensor, _ = pt_model(x_tensor)
            preds_gpu = preds_gpu_tensor.squeeze(0).cpu().numpy()

        diff_seq = np.abs(preds_gpu[need] - preds_step[need])
        max_d = np.max(diff_seq)
        mean_d = np.mean(diff_seq)
        max_step_diff = max(max_step_diff, max_d)

        acc_gpu.add(targets, preds_gpu, mask)
        acc_step.add(targets, preds_step, mask)
        print(f"Seq {seq_idx} (ID={seq_id}): max |GPU - Step| = {max_d:.2e}, mean = {mean_d:.2e}")

    res_gpu = acc_gpu.result()
    res_step = acc_step.result()
    score_diff = abs(res_gpu["weighted_pearson"] - res_step["weighted_pearson"])

    print("\n" + "-" * 65)
    print(f"Parity Results across {num_sequences} sequences:")
    print(f"  GPU Score  (GlobalAccumulator) : WP = {res_gpu['weighted_pearson']:.6f}")
    print(f"  Step Score (GlobalAccumulator) : WP = {res_step['weighted_pearson']:.6f}")
    print(f"  Score Absolute Difference      : {score_diff:.6f}")
    print(f"  Max Per-Step Difference        : {max_step_diff:.2e} (Tolerance: ~1e-4)")

    passed = (max_step_diff < 1.5e-4) and (score_diff < 1e-4)
    print(f"Status: {'PASSED [OK]' if passed else 'FAILED [X]'}")
    print("=" * 65)
    return passed


def run_gpu_full_eval(
    checkpoint_path: Path,
    valid_path: Path,
    num_sequences: int = -1,
    device: str = "cuda",
):
    """Evaluates checkpoint across validation dataset on GPU accumulating into GlobalAccumulator."""
    dev = torch.device(device)
    pt_model = load_pytorch_model(checkpoint_path, dev)
    pf = pq.ParquetFile(str(valid_path))
    total_seqs = pf.num_row_groups
    num_eval = total_seqs if num_sequences < 0 else min(num_sequences, total_seqs)

    acc_scored = GlobalAccumulator()
    acc_all = GlobalAccumulator()

    read_cols = FEATURE_COLUMNS + list(TARGET_COLUMNS) + ["need_prediction", "is_scored"]
    t0 = time.time()

    with torch.no_grad():
        for rg in range(num_eval):
            table = pf.read_row_group(rg, columns=read_cols)
            df = table.to_pandas()
            raw_feats = np.ascontiguousarray(df[FEATURE_COLUMNS].to_numpy(dtype=np.float32))
            targets = np.ascontiguousarray(df[list(TARGET_COLUMNS)].to_numpy(dtype=np.float32))
            need = df["need_prediction"].to_numpy(dtype=bool)
            scored = df["is_scored"].to_numpy(dtype=bool) if "is_scored" in df.columns else np.ones(20000, bool)

            feats = extract_32_features_sequence(raw_feats) if pt_model.input_dim > 112 else raw_feats
            x_gpu = torch.from_numpy(feats).unsqueeze(0).to(dev)
            preds_gpu_tensor, _ = pt_model(x_gpu)
            preds_gpu = preds_gpu_tensor.squeeze(0).cpu().numpy()

            acc_scored.add(targets, preds_gpu, need & scored)
            acc_all.add(targets, preds_gpu, need)

            if (rg + 1) % 200 == 0 or (rg + 1) == num_eval:
                elapsed = time.time() - t0
                print(f"[{rg+1}/{num_eval}] Elapsed: {elapsed:.1f}s ({(rg+1)/elapsed:.1f} seq/s)")

    res_scored = acc_scored.result()
    res_all = acc_all.result()

    print("\n" + "=" * 65)
    print(f"  FULL VALIDATION RESULTS ({num_eval} SEQUENCES)")
    print("=" * 65)
    print(f"Official Mask (need & is_scored): WP = {res_scored['weighted_pearson']:.6f}")
    print(f"  t0: {res_scored['t0']:.6f}, t1: {res_scored['t1']:.6f}")
    print(f"  Scored rows: {res_scored['selected_rows']:,}")
    print(f"\nUnfiltered Mask (need_prediction): WP = {res_all['weighted_pearson']:.6f}")
    print(f"  t0: {res_all['t0']:.6f}, t1: {res_all['t1']:.6f}")
    print(f"  Scored rows: {res_all['selected_rows']:,}")
    print(f"\nDelta vs Official Baseline (0.617052): {res_scored['weighted_pearson'] - 0.617052:+.6f}")
    print("=" * 65)
    return res_scored, res_all


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation Suite for Wunder RNN Challenge")
    parser.add_argument("--mode", type=str, choices=["parity", "latency", "gpu_full"], default="parity")
    parser.add_argument("--checkpoint", type=str, default=str(ROOT_DIR / "checkpoints_5epoch" / "best_model_5ep.pt"))
    parser.add_argument("--solution", type=str, default=str(ROOT_DIR / "submission_5epoch" / "solution.py"))
    parser.add_argument("--valid_path", type=str, default=str(ROOT_DIR / "datasets" / "valid.parquet"))
    parser.add_argument("--num_sequences", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--core_id", type=int, default=0)
    args = parser.parse_args()

    if args.mode == "parity":
        run_parity_check(Path(args.checkpoint), Path(args.solution), Path(args.valid_path), args.num_sequences, args.device)
    elif args.mode == "latency":
        run_latency_benchmark(Path(args.solution), Path(args.valid_path), core_id=args.core_id)
    elif args.mode == "gpu_full":
        run_gpu_full_eval(Path(args.checkpoint), Path(args.valid_path), args.num_sequences, args.device)
