"""ONNX Export and Numerical Parity Verification for ScaledWunderGRU.

Exports a trained PyTorch ScaledWunderGRU checkpoint to an optimized ONNX model
matching the competition inference specification:
  Inputs : ['features', 'hidden_0', 'hidden_1']
  Outputs: ['prediction', 'next_hidden_0', 'next_hidden_1']
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils import TARGET_COLUMNS
from src.features import TOTAL_FEATURES
from src.model import ScaledWunderGRU


class ScaledWunderGRUONNXWrapper(nn.Module):
    """ONNX export wrapper that unbinds multi-layer hidden states into individual layer tensors."""

    def __init__(self, model: ScaledWunderGRU):
        super().__init__()
        self.model = model
        self.num_layers = model.num_layers
        self.hidden_dim = model.hidden_dim

    def forward(
        self,
        features: torch.Tensor,
        hidden_0: torch.Tensor,
        hidden_1: torch.Tensor,
    ):
        # Concatenate individual layer hidden states along layer dimension
        hidden = torch.cat([hidden_0, hidden_1], dim=0)  # (2, 1, hidden_dim)
        pred, next_hidden = self.model(features, hidden)
        return pred, next_hidden[0:1], next_hidden[1:2]


def export_onnx(
    checkpoint_path: str,
    output_path: str,
    opset_version: int = 17,
):
    checkpoint_p = Path(checkpoint_path)
    if not checkpoint_p.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_p}")

    output_p = Path(output_path)
    output_p.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint from: {checkpoint_p}")
    checkpoint = torch.load(checkpoint_p, map_location="cpu")
    model_config = checkpoint.get("model_config", {})
    input_dim = model_config.get("input_dim", TOTAL_FEATURES)
    hidden_dim = model_config.get("hidden_dim", 128)
    num_layers = model_config.get("num_layers", 2)
    output_dim = model_config.get("output_dim", len(TARGET_COLUMNS))
    use_moe = model_config.get("use_moe", True)

    print(f"Instantiating ScaledWunderGRU:")
    print(f"  Input Dim  : {input_dim}")
    print(f"  Hidden Dim : {hidden_dim}")
    print(f"  Num Layers : {num_layers}")
    print(f"  Use MoE    : {use_moe}")

    base_model = ScaledWunderGRU(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_dim=output_dim,
        dropout=0.0,  # Eval mode
        use_moe=use_moe,
    )
    base_model.load_state_dict(checkpoint["model_state_dict"])
    base_model.eval()

    wrapper = ScaledWunderGRUONNXWrapper(base_model)
    wrapper.eval()

    # Create dummy inputs for tracing
    dummy_features = torch.randn(1, 1, input_dim, dtype=torch.float32)
    dummy_h0 = torch.zeros(1, 1, hidden_dim, dtype=torch.float32)
    dummy_h1 = torch.zeros(1, 1, hidden_dim, dtype=torch.float32)

    input_names = ["features", "hidden_0", "hidden_1"]
    output_names = ["prediction", "next_hidden_0", "next_hidden_1"]

    dynamic_axes = {
        "features": {1: "seq_len"},
        "prediction": {1: "seq_len"},
    }

    print(f"Exporting to ONNX: {output_p} (Opset {opset_version})...")
    torch.onnx.export(
        wrapper,
        (dummy_features, dummy_h0, dummy_h1),
        str(output_p),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
    )

    # 1. Check ONNX Model Structure
    onnx_model = onnx.load(str(output_p))
    onnx.checker.check_model(onnx_model)
    file_size_mb = output_p.stat().st_size / (1024 * 1024)
    print(f"[SUCCESS] ONNX model exported cleanly! File size: {file_size_mb:.2f} MB (Budget: <= 20 MB)")

    # 2. Check Parity with ONNX Runtime
    print("\nVerifying numerical parity between PyTorch and ONNX Runtime...")
    ort_options = ort.SessionOptions()
    ort_options.intra_op_num_threads = 1
    ort_options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(output_p), sess_options=ort_options, providers=["CPUExecutionProvider"])

    test_feat = np.random.randn(1, 1, input_dim).astype(np.float32)
    test_h0 = np.random.randn(1, 1, hidden_dim).astype(np.float32)
    test_h1 = np.random.randn(1, 1, hidden_dim).astype(np.float32)

    with torch.no_grad():
        pt_pred, pt_nh0, pt_nh1 = wrapper(
            torch.from_numpy(test_feat),
            torch.from_numpy(test_h0),
            torch.from_numpy(test_h1),
        )

    ort_pred, ort_nh0, ort_nh1 = session.run(
        None,
        {
            "features": test_feat,
            "hidden_0": test_h0,
            "hidden_1": test_h1,
        },
    )

    max_diff_pred = np.max(np.abs(pt_pred.numpy() - ort_pred))
    max_diff_h0 = np.max(np.abs(pt_nh0.numpy() - ort_nh0))
    max_diff_h1 = np.max(np.abs(pt_nh1.numpy() - ort_nh1))

    print(f"Max absolute discrepancy:")
    print(f"  Prediction  : {max_diff_pred:.2e}")
    print(f"  Next Hidden0: {max_diff_h0:.2e}")
    print(f"  Next Hidden1: {max_diff_h1:.2e}")

    assert max_diff_pred < 1e-4, f"Parity check failed on prediction: {max_diff_pred}"
    assert max_diff_h0 < 1e-4, f"Parity check failed on hidden_0: {max_diff_h0}"
    assert max_diff_h1 < 1e-4, f"Parity check failed on hidden_1: {max_diff_h1}"
    print("[SUCCESS] Full numerical parity verified!")


def main():
    parser = argparse.ArgumentParser(description="Export ScaledWunderGRU to ONNX")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(ROOT_DIR / "checkpoints" / "best_model.pt"),
        help="Path to trained PyTorch checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(ROOT_DIR / "submission" / "model.onnx"),
        help="Output path for ONNX model",
    )
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    args = parser.parse_args()

    export_onnx(args.checkpoint, args.output, args.opset)


if __name__ == "__main__":
    main()
