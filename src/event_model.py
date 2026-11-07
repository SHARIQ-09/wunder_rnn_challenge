"""Event-clock GRU: memory ticks only on i0 price-update rows; a light head predicts on every row.

Targets change only on i0 price updates, so the GRU sees the sequence on that clock (~20-30% of rows,
so ~3-5x longer memory per step and ~4x cheaper on average). Information that arrives between updates
(volume changes, trades, elapsed time) enters through a small per-row path, so predictions still move
between updates.

Inputs (all causal):
- event rows: "slim" features of the row + change in i0 volumes since the previous event (22)
  + log1p(steps since previous event) + log1p(trades since previous event)          -> EV_DIM
- every row:  raw row (112) + change in i0 volumes since the last event (22)
  + log1p(steps since last event) + log1p(trades since last event)                  -> ROW_DIM
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.feature_sets import build_features, feature_dim

N_EV_EXTRA = 24
EV_DIM = feature_dim("slim") + N_EV_EXTRA
ROW_DIM = 112 + 24


def event_inputs(raw: np.ndarray):
    """(T, 112) raw -> ev_x (E, EV_DIM), row_x (T, ROW_DIM), ev_index (T,) row -> index of its latest event."""
    T = raw.shape[0]
    x = raw.astype(np.float64)
    upd = np.ones(T, dtype=bool)
    upd[1:] = np.any(x[1:, 0:22] != x[:-1, 0:22], axis=1)
    trade = np.any(x[:, 48:52] != x[:, 48:49], axis=1)
    ev_pos = np.flatnonzero(upd)
    ev_index = np.cumsum(upd) - 1
    ctr = np.cumsum(trade)                                    # trades up to and including each row

    vol = x[:, 22:44]
    prev_pos = np.r_[ev_pos[0], ev_pos[:-1]]
    ev_extra = np.column_stack([vol[ev_pos] - vol[prev_pos],
                                np.log1p(ev_pos - prev_pos),
                                np.log1p(ctr[ev_pos] - ctr[prev_pos])])
    ev_x = np.hstack([build_features(raw, "slim")[ev_pos], ev_extra.astype(np.float32)])

    last = ev_pos[ev_index]
    row_x = np.hstack([raw, (vol - vol[last]).astype(np.float32),
                       np.log1p(np.arange(T) - last)[:, None].astype(np.float32),
                       np.log1p(ctr - ctr[last])[:, None].astype(np.float32)])
    return np.ascontiguousarray(ev_x, dtype=np.float32), np.ascontiguousarray(row_x, dtype=np.float32), ev_index


class EventGRU(nn.Module):
    def __init__(self, ev_dim: int = EV_DIM, row_dim: int = ROW_DIM, hidden_dim: int = 128,
                 num_layers: int = 2, row_proj: int = 48, dropout: float = 0.0, clip: float = 10.0):
        super().__init__()
        self.hidden_dim, self.num_layers, self.clip = hidden_dim, num_layers, clip
        self.register_buffer("ev_mean", torch.zeros(ev_dim)); self.register_buffer("ev_std", torch.ones(ev_dim))
        self.register_buffer("row_mean", torch.zeros(row_dim)); self.register_buffer("row_std", torch.ones(row_dim))
        self.input_proj = nn.Sequential(nn.Linear(ev_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        self.gru_layers = nn.ModuleList([nn.GRU(hidden_dim, hidden_dim, batch_first=True) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.row_path = nn.Sequential(nn.Linear(row_dim, row_proj), nn.GELU())
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim + row_proj), nn.Linear(hidden_dim + row_proj, hidden_dim),
                                  nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 2))

    def forward(self, ev_x: torch.Tensor, row_x: torch.Tensor, ev_index: torch.Tensor) -> torch.Tensor:
        """ev_x (B, E, ev_dim) padded at the end, row_x (B, T, row_dim), ev_index (B, T) long -> (B, T, 2)."""
        z = torch.clamp((ev_x - self.ev_mean) / self.ev_std, -self.clip, self.clip)
        h = self.input_proj(z)
        for i, gru in enumerate(self.gru_layers):
            out, _ = gru(h)
            h = self.norms[i](out) if i == 0 else self.norms[i](out + h)
        mem = torch.gather(h, 1, ev_index.unsqueeze(-1).expand(-1, -1, h.size(-1)))
        r = self.row_path(torch.clamp((row_x - self.row_mean) / self.row_std, -self.clip, self.clip))
        return self.head(torch.cat([mem, r], dim=-1))
