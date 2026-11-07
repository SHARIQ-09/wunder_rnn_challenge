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


class LongMemory(nn.Module):
    """Bank of fixed-timescale EMAs over learned input mixes: e_t = (1 - a) e_{t-1} + a u_t, a = 1 / tau.

    The recurrence is linear, so a whole chunk is computed in parallel with exact block matrices
    (no python loop over time, all decay powers <= 1). EMAs are bias-corrected by the running weight
    w_t = 1 - prod(1 - a), so a fresh state starts from the first value instead of zero.
    Output per step: [e (S*K), u - e (S*K)]. State per sequence: S*K EMAs + S weights.
    """

    BLOCK = 64

    def __init__(self, mem_dim: int, scales: Tuple[float, ...]):
        super().__init__()
        self.K, self.S = mem_dim, len(scales)
        a = torch.tensor([1.0 / s for s in scales], dtype=torch.float64)
        r = 1.0 - a
        b = self.BLOCK
        i = torch.arange(b, dtype=torch.float64)
        diff = i[:, None] - i[None, :]
        local = torch.where(diff >= 0, a[:, None, None] * r[:, None, None] ** diff.clamp(min=0), torch.zeros(()))
        self.register_buffer("log_r", torch.log1p(-a).float(), persistent=False)  # float64 log: 1 - 1e-4 is coarse in float32
        self.register_buffer("local", local.float(), persistent=False)                       # (S, b, b)
        self.register_buffer("rpow", (r[:, None] ** (i[None, :] + 1)).float(), persistent=False)  # (S, b): r^(i+1)

    @property
    def state_dim(self) -> int:
        return self.S * self.K + self.S

    def forward(self, u: torch.Tensor, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """u (B, T, K), state (B, >= state_dim) -> features (B, T, 2*S*K), new state (B, state_dim)."""
        B, T, K = u.shape
        S, b = self.S, self.BLOCK
        u = u.float()
        e0 = state[:, :S * K].float().view(B, S, K)
        w0 = state[:, S * K:S * K + S].float()
        nb = -(-T // b)
        U = torch.nn.functional.pad(u, (0, 0, 0, nb * b - T)).view(B, nb, b, K)
        loc = torch.einsum("sij,bnjk->bsnik", self.local, U)                                 # (B, S, nb, b, K)
        # Carry into each block: c_n = loc_n[-1] + R c_{n-1}, c_{-1} = e0, R = r^b.
        n = torch.arange(nb, device=u.device, dtype=torch.float32)
        lr = self.log_r[:, None, None]                                               # (S, 1, 1)
        dn = n[:, None] - n[None, :]
        M = torch.where(dn >= 0, torch.exp(lr * b * dn.clamp(min=0)), torch.zeros((), device=u.device))  # (S, nb, nb)
        c = torch.einsum("snm,bsmk->bsnk", M, loc[:, :, :, -1, :])
        c = c + torch.exp(lr * b * (n[None, :, None] + 1))[None] * e0[:, :, None, :]
        prev = torch.cat([e0[:, :, None, :], c[:, :, :-1, :]], dim=2)                         # carry entering block n
        y = loc + self.rpow[None, :, None, :, None] * prev[:, :, :, None, :]
        y = y.reshape(B, S, nb * b, K)[:, :, :T]                                              # (B, S, T, K)
        # Running weight: w_t = 1 - (1 - w0) r^(t+1)
        t = torch.arange(T, device=u.device, dtype=torch.float32)
        lt = self.log_r[:, None] * (t[None, :] + 1)                                          # (S, T)
        w = w0[:, :, None] * torch.exp(lt)[None] - torch.expm1(lt)[None]                       # (B, S, T), no 1 - r cancellation
        e = y / w[..., None].clamp(min=1e-6)
        new_state = torch.cat([y[:, :, -1, :].reshape(B, S * K), w[:, :, -1]], dim=1)
        e = e.permute(0, 2, 1, 3).reshape(B, T, S * K)
        dev = u.repeat(1, 1, S) - e
        return torch.cat([e, dev], dim=-1), new_state


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
        mem_dim: int = 0,
        mem_scales: Tuple[float, ...] = (20.0, 200.0, 2000.0, 10000.0),
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.use_moe = use_moe
        self.mem_dim = mem_dim

        # Optional long memory: learned input mixes -> fixed-timescale EMAs -> added to the input projection.
        # Its state rides in one extra hidden slot, so hidden is (num_layers + 1, batch, hidden_dim).
        if mem_dim > 0:
            self.mem_in = nn.Linear(input_dim, mem_dim)
            self.mem = LongMemory(mem_dim, tuple(mem_scales))
            if self.mem.state_dim > hidden_dim:
                raise ValueError(f"memory state {self.mem.state_dim} does not fit hidden_dim {hidden_dim}")
            self.mem_out = nn.Sequential(nn.LayerNorm(2 * len(mem_scales) * mem_dim),
                                         nn.Linear(2 * len(mem_scales) * mem_dim, hidden_dim))

        # Per-feature input standardization (identity by default; set via set_norm_stats).
        # Baked into the ONNX graph, so solution.py keeps feeding raw 144-dim features.
        self.register_buffer("feat_mean", torch.zeros(input_dim))
        self.register_buffer("feat_std", torch.ones(input_dim))
        self.register_buffer("norm_clip", torch.tensor(0.0))  # 0 disables clipping

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

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Accepts pre-normalization checkpoints by filling missing norm buffers with identity."""
        state_dict = dict(state_dict)
        for key in ("feat_mean", "feat_std", "norm_clip"):
            if key not in state_dict:
                state_dict[key] = getattr(self, key).detach().clone()
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    @property
    def has_norm(self) -> bool:
        return bool(
            torch.any(self.feat_mean != 0).item() or torch.any(self.feat_std != 1).item()
        )

    @torch.no_grad()
    def set_norm_stats(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        clip: float = 0.0,
        fold_into_input: bool = False,
    ) -> None:
        """Installs feature standardization.

        With fold_into_input=True the first Linear layer is rescaled so the network
        computes exactly the same function as before (useful to warm-start a
        checkpoint trained on raw features): W' = W * std, b' = b + W @ mean.
        """
        mean = mean.to(self.feat_mean.device, torch.float32)
        std = std.to(self.feat_std.device, torch.float32)
        if fold_into_input:
            if self.has_norm:
                raise RuntimeError("Model already has normalization; refusing to fold twice.")
            lin = self.input_proj[0]
            w = lin.weight.data.clone()
            lin.bias.data += w @ mean
            lin.weight.data = w * std.unsqueeze(0)
        self.feat_mean.copy_(mean)
        self.feat_std.copy_(std)
        self.norm_clip.fill_(float(clip))

    def init_hidden(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Initializes hidden states of shape (num_layers, batch_size, hidden_dim)."""
        if device is None:
            device = next(self.parameters()).device
        n_slots = self.num_layers + (1 if self.mem_dim > 0 else 0)
        return torch.zeros(n_slots, batch_size, self.hidden_dim, device=device)

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

        # Standardize features, optionally clip outliers, then project
        x = (x - self.feat_mean) / self.feat_std
        if float(self.norm_clip) > 0:
            x = torch.clamp(x, -float(self.norm_clip), float(self.norm_clip))
        next_hiddens: List[torch.Tensor] = []
        if self.mem_dim > 0:
            with torch.autocast(device_type=x.device.type, enabled=False):
                m, mem_state = self.mem(self.mem_in(x.float()), hidden[-1])
            h_current = self.input_proj[1](self.input_proj[0](x) + self.mem_out(m))
        else:
            h_current = self.input_proj(x)

        for i, gru in enumerate(self.gru_layers):
            layer_hidden = hidden[i:i+1].contiguous()
            out, next_h = gru(h_current, layer_hidden)
            next_hiddens.append(next_h)

            if i == 0:
                h_current = self.drop(self.norms[i](out))
            else:
                # Inter-layer residual connection
                h_current = self.drop(self.norms[i](out + h_current))

        if self.mem_dim > 0:
            # Keep the whole state in float32 (AMP would round slow EMAs at every chunk boundary).
            next_hiddens = [h.float() for h in next_hiddens]
            pad = mem_state.new_zeros(batch_size, self.hidden_dim - mem_state.size(1))
            next_hiddens.append(torch.cat([mem_state, pad], dim=1).unsqueeze(0))
        all_next_hidden = torch.cat(next_hiddens, dim=0)  # (num_layers [+1 memory], batch, hidden)
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


class BranchGRU(ScaledWunderGRU):
    """Input-specialist GRUs (price, volume, trade, a0-a7) joined by a fusion GRU, on the raw_ev_ema feature layout.

    Each branch sees only its own columns (raw, one-step diff, two EMA gaps; price and volume also get their
    update-event features). Branch outputs are concatenated, projected and passed through the fusion GRU, so price
    and volume can interact over time. State: one hidden slot per GRU, (5, batch, hidden_dim).
    """

    FAMILIES = {
        "price": [*range(0, 22), *range(52, 74)],
        "volume": [*range(22, 44), *range(74, 96)],
        "trade": [*range(44, 52), *range(96, 104)],
        "a": [*range(104, 112)],
    }
    EVENTS = {"price": [*range(112, 116)], "volume": [*range(116, 120)], "trade": [], "a": []}
    # v144 layout: raw columns plus engineered f1-f32 at 112 + (f - 1), assigned to the family they are computed from.
    V144_ENG = {
        "price": [112 + f - 1 for f in (1, 2, 3, 4, 10, 11, 16, 17, 18, 19, 20, 25, 26, 27, 28, 31)],
        "volume": [112 + f - 1 for f in (5, 6, 7, 8, 9, 12, 21, 22, 23, 24, 32)],
        "trade": [112 + f - 1 for f in (13, 14, 15, 29, 30)],
        "a": [],
    }

    def __init__(self, input_dim: int = 456, hidden_dim: int = 80, output_dim: int = 2, dropout: float = 0.05):
        if input_dim not in (144, 456):
            raise ValueError("BranchGRU expects the v144 (144) or raw_ev_ema (456) layout")
        super().__init__(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=len(self.FAMILIES) + 1,
                         output_dim=output_dim, dropout=dropout, use_moe=False)
        del self.input_proj, self.gru_layers, self.norms, self.head
        self.names = list(self.FAMILIES)
        self.branch_proj, self.branch_gru, self.branch_norm = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for k, name in enumerate(self.names):
            cols = self.FAMILIES[name]
            if input_dim == 144:
                idx = cols + self.V144_ENG[name]
            else:
                idx = cols + self.EVENTS[name] + [c + 120 for c in cols] + [c + 232 for c in cols] + [c + 344 for c in cols]
            self.register_buffer(f"idx_{k}", torch.tensor(idx, dtype=torch.long), persistent=False)
            self.branch_proj.append(nn.Sequential(nn.Linear(len(idx), hidden_dim), nn.LayerNorm(hidden_dim)))
            self.branch_gru.append(nn.GRU(hidden_dim, hidden_dim, batch_first=True))
            self.branch_norm.append(nn.LayerNorm(hidden_dim))
        self.fusion_proj = nn.Sequential(nn.Linear(len(self.names) * hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        self.fusion_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout if dropout > 0 else 0.0),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if hidden is None:
            hidden = self.init_hidden(x.size(0), device=x.device)
        x = (x - self.feat_mean) / self.feat_std
        if float(self.norm_clip) > 0:
            x = torch.clamp(x, -float(self.norm_clip), float(self.norm_clip))
        outs, next_h = [], []
        for k in range(len(self.names)):
            z = self.branch_proj[k](x[..., getattr(self, f"idx_{k}")])
            out, h = self.branch_gru[k](z, hidden[k:k + 1].contiguous())
            outs.append(self.drop(self.branch_norm[k](out + z)))
            next_h.append(h)
        f = self.fusion_proj(torch.cat(outs, dim=-1))
        out, h = self.fusion_gru(f, hidden[-1:].contiguous())
        next_h.append(h)
        return self.head(self.drop(self.fusion_norm(out + f))), torch.cat(next_h, dim=0)


class BranchClsGRU(BranchGRU):
    """BranchGRU with a feed-forward layer before each branch GRU and a classification head.

    Targets sit on per-sequence grids (multiples of UNITS, runs/decode/units.log). The head outputs one distribution per
    target over the union of those grid values within the metric clip; the prediction is its expected value, so the
    metric loss and all evaluation code work unchanged. ce_loss adds cross-entropy on the nearest class.
    """

    UNITS = (0.65687, 0.24133, 0.46854, 0.12543)

    def __init__(self, input_dim: int = 456, hidden_dim: int = 80, output_dim: int = 2, dropout: float = 0.05):
        super().__init__(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, dropout=dropout)
        clip = 2.0
        vals = {0.0, clip, -clip}
        for u in self.UNITS:
            n = int(clip / u)
            vals.update(k * u for k in range(-n, n + 1) if k != 0)
        self.register_buffer("cls_values", torch.tensor(sorted(vals), dtype=torch.float32))
        self.n_cls = len(vals)
        for k in range(len(self.names)):
            n_in = len(getattr(self, f"idx_{k}"))
            self.branch_proj[k] = nn.Sequential(nn.Linear(n_in, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
                                                nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        self.head[-1] = nn.Linear(hidden_dim, output_dim * self.n_cls)
        self.cls_logits = None

    def forward(self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        out, h = super().forward(x, hidden)
        logits = out.view(out.size(0), out.size(1), -1, self.n_cls)
        self.cls_logits = logits if self.training else None
        return (torch.softmax(logits, dim=-1) * self.cls_values).sum(-1), h

    def ce_loss(self, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Cross-entropy on the nearest class, weighted by |t| + 0.1 so rows with t = 0 still teach the zero class."""
        y = torch.clamp(targets[mask], -2.0, 2.0)
        if y.numel() == 0:
            return self.cls_logits.sum() * 0.0
        cls = (y.unsqueeze(-1) - self.cls_values).abs().argmin(-1)
        ce = nn.functional.cross_entropy(self.cls_logits[mask].reshape(-1, self.n_cls), cls.reshape(-1), reduction="none")
        w = y.abs().reshape(-1) + 0.1
        return (w * ce).sum() / w.sum()


# Alias for backward compatibility
WunderGRU = ScaledWunderGRU


class HybridLoss(nn.Module):
    """Metric-Aligned Loss combining Target-Wise |t|-Weighted MSE and Pooled Weighted Pearson.

    Strictly matches the competition evaluation contract:
    - Targets clipped to [-2.0, 2.0].
    - Weights w = |clipped_targets|.
    - Target-wise Weighted MSE penalizes scale and squared error.
    - Pooled Weighted Pearson directly maximizes global correlation across all batch steps.
    """

    def __init__(
        self,
        clip_val: float = METRIC_CLIP,
        mse_weight: float = 1.0,
        pearson_weight: float = 1.0,
        t0_weight: float = 0.5,
        t1_weight: float = 0.5,
        aligned: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.clip_val = clip_val
        self.mse_weight = mse_weight
        self.pearson_weight = pearson_weight
        self.t0_weight = t0_weight
        self.t1_weight = t1_weight
        self.aligned = aligned
        self.eps = eps

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        row_weight: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """row_weight: optional (B, T) multiplier on the |t| weights (aligned mode only)."""
        batch_size, seq_len, n_targets = predictions.shape

        if not self.aligned:
            # Legacy unaligned loss mode: unclipped targets in MSE + unaligned per-chunk Pearson
            clipped_t = torch.clamp(targets, -self.clip_val, self.clip_val)
            w_mse = torch.abs(clipped_t) + 1e-4
            if mask is not None:
                p_m = predictions[mask]
                t_m = targets[mask]
                w_m = w_mse[mask]
            else:
                p_m = predictions.reshape(-1)
                t_m = targets.reshape(-1)
                w_m = w_mse.reshape(-1)

            if p_m.numel() == 0:
                zero_l = torch.tensor(0.0, device=predictions.device, requires_grad=True)
                return {"loss": zero_l, "weighted_mse": zero_l, "pearson_loss": zero_l}

            weighted_mse = torch.sum(w_m * ((p_m - t_m) ** 2)) / (torch.sum(w_m) + self.eps)

            p_flat = predictions.reshape(-1, n_targets)
            t_flat = targets.reshape(-1, n_targets)
            if mask is not None:
                m_flat = mask.reshape(-1)
                p_flat = p_flat[m_flat]
                t_flat = t_flat[m_flat]

            if p_flat.size(0) >= 10:
                t_c = torch.clamp(t_flat, -self.clip_val, self.clip_val)
                w_p = torch.abs(t_c)
                corrs = []
                for side in range(n_targets):
                    w = w_p[:, side]
                    w_tot = w.sum()
                    if w_tot < self.eps:
                        continue
                    y_s = t_c[:, side]
                    p_s = p_flat[:, side]
                    mean_y = (w * y_s).sum() / w_tot
                    mean_p = (w * p_s).sum() / w_tot
                    yc = y_s - mean_y
                    pc = p_s - mean_p
                    cov = (w * yc * pc).sum() / w_tot
                    var_y = (w * (yc ** 2)).sum() / w_tot
                    var_p = (w * (pc ** 2)).sum() / w_tot
                    std_prod = torch.sqrt(var_y * var_p + self.eps)
                    corrs.append(torch.clamp(cov / std_prod, -1.0, 1.0))

                pearson_loss = 1.0 - torch.stack(corrs).mean() if len(corrs) == n_targets else torch.tensor(0.0, device=predictions.device)
            else:
                pearson_loss = torch.tensor(0.0, device=predictions.device)

            total_loss = self.mse_weight * weighted_mse + self.pearson_weight * pearson_loss
            return {"loss": total_loss, "weighted_mse": weighted_mse, "pearson_loss": pearson_loss}

        # Metric-Aligned Loss mode:
        clipped_t = torch.clamp(targets, -self.clip_val, self.clip_val)
        weights = torch.abs(clipped_t)
        if row_weight is not None:
            weights = weights * row_weight.unsqueeze(-1).to(weights.dtype)

        if mask is not None:
            p_flat = predictions[mask]      # (N, 2)
            y_flat = clipped_t[mask]       # (N, 2)
            w_flat = weights[mask]         # (N, 2)
        else:
            p_flat = predictions.reshape(-1, n_targets)
            y_flat = clipped_t.reshape(-1, n_targets)
            w_flat = weights.reshape(-1, n_targets)

        if p_flat.size(0) < 10:
            zero_loss = torch.tensor(0.0, device=predictions.device, requires_grad=True)
            return {
                "loss": zero_loss,
                "weighted_mse": zero_loss,
                "pearson_loss": zero_loss,
            }

        # ── 1. Target-Wise |t|-Weighted MSE ──────────────────────────
        w0, w1 = w_flat[:, 0], w_flat[:, 1]
        tot0 = w0.sum() + self.eps
        tot1 = w1.sum() + self.eps

        mse0 = torch.sum(w0 * ((p_flat[:, 0] - y_flat[:, 0]) ** 2)) / tot0
        mse1 = torch.sum(w1 * ((p_flat[:, 1] - y_flat[:, 1]) ** 2)) / tot1
        weighted_mse = self.t0_weight * mse0 + self.t1_weight * mse1

        # ── 2. Pooled Batch-Level Weighted Pearson Correlation ───────
        corrs = []
        for side, w_s, tot in [(0, w0, tot0), (1, w1, tot1)]:
            y_s = y_flat[:, side]
            p_s = p_flat[:, side]

            mean_y = (w_s * y_s).sum() / tot
            mean_p = (w_s * p_s).sum() / tot

            yc = y_s - mean_y
            pc = p_s - mean_p

            cov = (w_s * yc * pc).sum() / tot
            var_y = (w_s * (yc ** 2)).sum() / tot
            var_p = (w_s * (pc ** 2)).sum() / tot

            std_prod = torch.sqrt(var_y * var_p + self.eps)
            r = cov / std_prod
            corrs.append(torch.clamp(r, -1.0, 1.0))

        pearson_loss = self.t0_weight * (1.0 - corrs[0]) + self.t1_weight * (1.0 - corrs[1])
        total_loss = self.mse_weight * weighted_mse + self.pearson_weight * pearson_loss

        return {
            "loss": total_loss,
            "weighted_mse": weighted_mse,
            "pearson_loss": pearson_loss,
            "r0": corrs[0],
            "r1": corrs[1],
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
