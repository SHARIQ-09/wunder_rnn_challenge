"""PyTorch Model Architectures for Alpha Connectome (Wunder RNN Challenge).

Includes:
- ScaledWunderGRU: 384-dim multi-layer stateful GRU with inter-layer residuals.
- HybridLoss: Weighted MSE + Batch-level Negative Pearson Correlation.
"""
from __future__ import annotations

from typing import Optional, Tuple, Dict, List
import torch
import torch.nn as nn

import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils import METRIC_CLIP
from src.features import TOTAL_FEATURES  # 128 features


class ScaledWunderGRU(nn.Module):
    """High-capacity stateful GRU with inter-layer residual connections.

    Supports:
    - Flexible input dimensions (128 engineered features or 112 raw features).
    - Scaled hidden dimension (default: 128 for guaranteed 32-min CPU inference).
    - Multi-layer recurrent processing with inter-layer LayerNorm residuals.
    - Fast 1-step online inference interface for competition submission.
    """

    def __init__(
        self,
        input_dim: int = TOTAL_FEATURES,  # 144
        hidden_dim: int = 128,
        num_layers: int = 2,
        output_dim: int = 2,
        dropout: float = 0.05,
        use_moe: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.use_moe = use_moe

        # Input projection into recurrent dimension with Pre-GRU LayerNorm
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Multi-layer GRU with inter-layer residual connections
        self.gru_layers = nn.ModuleList([
            nn.GRU(hidden_dim, hidden_dim, batch_first=True)
            for _ in range(num_layers)
        ])

        # Inter-layer LayerNorms and dropouts
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.drop = nn.Dropout(dropout if dropout > 0 else 0.0)

        # Output regression: Full-width GELU Head or MoE Dual Experts
        if use_moe:
            self.head_norm = nn.LayerNorm(hidden_dim)
            self.gate = nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.SiLU(),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            )
            self.head_quiet = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout if dropout > 0 else 0.0),
                nn.Linear(hidden_dim, output_dim),
            )
            self.head_shock = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout if dropout > 0 else 0.0),
                nn.Linear(hidden_dim, output_dim),
            )
        else:
            self.head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout if dropout > 0 else 0.0),
                nn.Linear(hidden_dim, output_dim),
            )

    def init_hidden(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Initializes hidden states of shape (num_layers, batch_size, hidden_dim)."""
        if device is None:
            device = next(self.parameters()).device
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass over sequence.

        Args:
            x: Shape (batch_size, seq_len, input_dim)
            hidden: Shape (num_layers, batch_size, hidden_dim) or None

        Returns:
            predictions: Shape (batch_size, seq_len, 2)
            next_hidden: Shape (num_layers, batch_size, hidden_dim)
        """
        batch_size = x.size(0)
        if hidden is None:
            hidden = self.init_hidden(batch_size, device=x.device)

        # Project input to hidden dimension
        h_current = self.input_proj(x)

        next_hiddens: List[torch.Tensor] = []
        for i, gru in enumerate(self.gru_layers):
            layer_hidden = hidden[i:i+1].contiguous()
            out, next_h = gru(h_current, layer_hidden)
            next_hiddens.append(next_h)

            if i == 0:
                h_current = self.drop(self.norms[i](out))
            else:
                # Inter-layer residual connection
                h_current = self.drop(self.norms[i](out + h_current))

        all_next_hidden = torch.cat(next_hiddens, dim=0)  # (num_layers, batch, hidden)
        if self.use_moe:
            h_norm = self.head_norm(h_current)
            gate_w = self.gate(h_norm)
            pred_quiet = self.head_quiet(h_norm)
            pred_shock = self.head_shock(h_norm)
            predictions = (1.0 - gate_w) * pred_quiet + gate_w * pred_shock
        else:
            predictions = self.head(h_current)

        return predictions, all_next_hidden

    def step(
        self,
        x_step: torch.Tensor,
        hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single-step inference for competition predict(data_point)."""
        if x_step.dim() == 2:
            x_step = x_step.unsqueeze(1)

        pred, next_hidden = self.forward(x_step, hidden)
        return pred.squeeze(1), next_hidden


# Alias for backward compatibility
WunderGRU = ScaledWunderGRU


class HybridLoss(nn.Module):
    """Hybrid Loss combining Weighted MSE and Batch-Level Negative Pearson Correlation.

    - Weighted MSE guarantees smooth convergence and penalizes prediction error.
    - Pearson component directly rewards relative ranking and inflection point timing.
    """

    def __init__(
        self,
        clip_val: float = METRIC_CLIP,
        mse_weight: float = 1.0,
        pearson_weight: float = 1.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.clip_val = clip_val
        self.mse_weight = mse_weight
        self.pearson_weight = pearson_weight
        self.eps = eps

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size, seq_len, n_targets = predictions.shape

        if mask is not None:
            if mask.dim() == 2:
                mask_expanded = mask.unsqueeze(-1).expand_as(predictions)
            else:
                mask_expanded = mask
            p = predictions[mask_expanded]
            t = targets[mask_expanded]
        else:
            p = predictions.reshape(-1)
            t = targets.reshape(-1)

        if p.numel() == 0:
            zero_loss = torch.tensor(0.0, device=predictions.device, requires_grad=True)
            return {"loss": zero_loss, "weighted_mse": zero_loss, "pearson_loss": zero_loss}

        # ── 1. Weighted MSE ──────────────────────────────────────────
        clipped_t = torch.clamp(t, -self.clip_val, self.clip_val)
        w_mse = torch.abs(clipped_t) + 1e-4
        weighted_mse = torch.sum(w_mse * ((p - t) ** 2)) / torch.sum(w_mse)

        # ── 2. Batch-Level Weighted Pearson Correlation ──────────────
        p_flat = predictions.reshape(-1, n_targets)
        t_flat = targets.reshape(-1, n_targets)

        if mask is not None:
            m_flat = mask.reshape(-1)
            p_flat = p_flat[m_flat]
            t_flat = t_flat[m_flat]

        if p_flat.size(0) >= 10:
            t_clipped = torch.clamp(t_flat, -self.clip_val, self.clip_val)
            w_p = torch.abs(t_clipped)  # (N, 2)

            corrs = []
            for side in range(n_targets):
                w = w_p[:, side]
                w_tot = w.sum()
                if w_tot < self.eps:
                    continue

                y_s = t_clipped[:, side]
                p_s = p_flat[:, side]

                mean_y = (w * y_s).sum() / w_tot
                mean_p = (w * p_s).sum() / w_tot

                yc = y_s - mean_y
                pc = p_s - mean_p

                cov = (w * yc * pc).sum() / w_tot
                var_y = (w * (yc ** 2)).sum() / w_tot
                var_p = (w * (pc ** 2)).sum() / w_tot

                std_prod = torch.sqrt(var_y * var_p + self.eps)
                r = cov / std_prod
                corrs.append(torch.clamp(r, -1.0, 1.0))

            if len(corrs) == n_targets:
                mean_corr = torch.stack(corrs).mean()
                pearson_loss = 1.0 - mean_corr
            else:
                pearson_loss = torch.tensor(0.0, device=predictions.device)
        else:
            pearson_loss = torch.tensor(0.0, device=predictions.device)

        total_loss = self.mse_weight * weighted_mse + self.pearson_weight * pearson_loss

        return {
            "loss": total_loss,
            "weighted_mse": weighted_mse,
            "pearson_loss": pearson_loss,
        }


# Aliases for backward compatibility
WeightedMSELoss = HybridLoss
CompositeLoss = HybridLoss


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Testing ScaledWunderGRU (src/model.py) on {device} ===")

    model = ScaledWunderGRU(
        input_dim=135,
        hidden_dim=384,
        num_layers=2,
        output_dim=2,
    ).to(device)

    params = sum(p.numel() for p in model.parameters())
    mb_size = (params * 4) / (1024 * 1024)
    print(f"Total Parameters     : {params:,}")
    print(f"Estimated Size (f32) : {mb_size:.2f} MB (Budget: <= 20 MB)")

    dummy_x = torch.randn(2, 2048, 135, device=device)
    dummy_t = torch.randn(2, 2048, 2, device=device)
    dummy_m = torch.ones(2, 2048, dtype=torch.bool, device=device)

    h0 = model.init_hidden(2, device)
    preds, h1 = model(dummy_x, h0)
    print(f"Forward Pass Shape   : {preds.shape}, Hidden Shape: {h1.shape}")

    criterion = HybridLoss(mse_weight=1.0, pearson_weight=0.5)
    loss_dict = criterion(preds, dummy_t, dummy_m)
    print(f"Loss Output: total={loss_dict['loss'].item():.4f}, mse={loss_dict['weighted_mse'].item():.4f}, pearson={loss_dict['pearson_loss'].item():.4f}")

    loss_dict["loss"].backward()
    print("[SUCCESS] ScaledWunderGRU and HybridLoss verified successfully!")
