"""Alternative causal feature sets for feature-ablation experiments (batch extraction only).

The raw columns are anonymized and pre-scaled (i0_p0..p21 are not sorted bid/ask levels), so these
sets avoid assumed order-book semantics and use only generic, causal transforms:
- "v144":  existing 112 raw (stationary) + 32 engineered (src/features.py), the current default.
- "raw":   the 112 raw columns unchanged.
- "raw_ev_ema": raw + update events + one-step diffs + deviations from EMAs (alpha 0.05, 0.005).
- "v144_ev_ema": v144 + the same event/diff/EMA block.
- "slim":  raw_ev_ema + 3 kept engineered trade features (money flow, EMA trade volume, EMA money flow).
- "pressure": slim + buy/sell pressure, activity bursts, liquidity + shocks, reversal + outside signal
  (uses frozen weights from datasets/pressure_weights.npz, see src/fit_pressure_weights.py).
- "momentum": raw_ev_ema + 36 nonlinear pressure-momentum features (scaled, trade-weighted, per-update,
  persistence) for buy, sell and net pressure; same frozen pressure weights.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.signal import lfilter

from src.features import extract_32_features_sequence

EMA_ALPHAS = (0.05, 0.005)
# Column blocks whose rows change together: i0 prices, i1 prices, i0 volumes, i1 volumes.
EVENT_BLOCKS = ((0, 22), (52, 74), (22, 44), (74, 96))


def _ema(x: np.ndarray, alpha: float) -> np.ndarray:
    return lfilter([alpha], [1.0, -(1.0 - alpha)], x, axis=0, zi=(1.0 - alpha) * x[:1])[0]


def _events(x: np.ndarray) -> np.ndarray:
    """Per block: updated-this-step flag and log1p(steps since last update). First row counts as an update."""
    L = x.shape[0]
    idx = np.arange(L)
    out = []
    for a, b in EVENT_BLOCKS:
        changed = np.ones(L, dtype=bool)
        changed[1:] = np.any(x[1:, a:b] != x[:-1, a:b], axis=1)
        last = np.maximum.accumulate(np.where(changed, idx, 0))
        out += [changed.astype(np.float64), np.log1p(idx - last)]
    return np.column_stack(out)


N_EV_EMA = 2 * len(EVENT_BLOCKS) + 112 * (1 + len(EMA_ALPHAS))


def _fill_ev_ema(x: np.ndarray, out: np.ndarray) -> None:
    """Writes events, one-step diffs and EMA deviations into the float32 view `out` (keeps peak memory low)."""
    ne = 2 * len(EVENT_BLOCKS)
    out[:, :ne] = _events(x)
    out[0, ne:ne + 112] = 0.0
    np.subtract(x[1:], x[:-1], out=out[1:, ne:ne + 112], casting="same_kind")
    for k, a in enumerate(EMA_ALPHAS):
        c = ne + 112 * (k + 1)
        out[:, c:c + 112] = x - _ema(x, a)


PRESSURE_WEIGHTS = Path(__file__).resolve().parents[1] / "datasets" / "pressure_weights.npz"
_PW = None


def _pressure_weights():
    global _PW
    if _PW is None:
        _PW = {k: v for k, v in np.load(PRESSURE_WEIGHTS).items()}
    return _PW


def _trade_keep(x: np.ndarray) -> np.ndarray:
    """The 3 engineered features kept from v144: money flow and EMAs (0.05) of trade volume and money flow."""
    dv = x[:, 48:52].sum(axis=1)
    mf = (x[:, 44:48] * x[:, 48:52]).sum(axis=1)
    return np.column_stack([mf, _ema(dv, 0.05), _ema(mf, 0.05)])


def _since(flag: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Steps since the flag was last true (flag[0] must be True, or counts from a virtual event at -1)."""
    last = np.maximum.accumulate(np.where(flag, idx, -1))
    return idx - last


def _rolling_count(flag: np.ndarray, n: int) -> np.ndarray:
    c = np.cumsum(flag, dtype=np.float64)
    out = c.copy()
    out[n:] -= c[:-n]
    return out


# Column layout of the "pressure" block (after raw_ev_ema 456 + 3 kept trade features).
PRESSURE_GROUPS = {"pressure": (0, 24), "bursts": (24, 31), "liquidity": (31, 41), "reversal": (41, 49)}
N_PRESSURE = 49


def _pressure(x: np.ndarray) -> np.ndarray:
    """Buy/sell pressure, activity bursts, liquidity + shocks, reversal + outside signal (all causal)."""
    w = _pressure_weights()
    L = x.shape[0]
    idx = np.arange(L)
    upd = np.ones(L, dtype=bool)
    upd[1:] = np.any(x[1:, 0:22] != x[:-1, 0:22], axis=1)
    trade = np.any(x[:, 48:52] != x[:, 48:49], axis=1)        # i0 trade volumes not all equal = a trade printed
    since_upd = _since(upd, idx)

    # 1. pressure: buy scores (B_t0, B_t1) and sell scores (S_t0, S_t1)
    Vz = (x[:, 22:44] - w["vmean"]) / w["vstd"]
    P4 = np.column_stack([Vz[:, :11] @ w["wB"].T, Vz[:, 11:] @ w["wS"].T])      # B0 B1 S0 S1
    net = P4[:, :2] - P4[:, 2:]
    share = P4[:, :2] / (np.abs(P4[:, :2]) + np.abs(P4[:, 2:]) + 0.1)
    at_upd = P4[np.maximum.accumulate(np.where(upd, idx, 0))]
    d_since = P4 - at_upd
    g1 = np.column_stack([P4, net, share, d_since, _ema(P4, 0.1), _ema(P4, 0.01), d_since * np.log1p(since_upd)[:, None]])

    # 2. activity bursts
    tf, uf = trade.astype(np.float64), upd.astype(np.float64)
    t10, t50 = _rolling_count(tf, 10), _rolling_count(tf, 50)
    u10, u50 = _rolling_count(uf, 10), _rolling_count(uf, 50)
    g2 = np.column_stack([t10, t50, np.log1p(_since(trade, idx)), u10, u50,
                          np.log((t10 / 10 + 0.01) / (t50 / 50 + 0.01)), np.log((u10 / 10 + 0.01) / (u50 / 50 + 0.01))])

    # 3. liquidity + shocks; pstep = scalar price move per step
    pstep = np.zeros(L)
    pstep[1:] = (np.diff(x[:, 0:22], axis=0) / w["dstd"]) @ w["pcomp"]
    mv = np.abs(pstep)
    mv_s, upd_s, tr_s = _ema(mv, 0.01), _ema(uf, 0.01), _ema(tf, 0.01)
    scale2 = lfilter([0.005], [1.0, -0.995], mv ** 2, zi=[0.995 * float(w["mv2_init"])])[0]
    prev_scale = np.sqrt(np.r_[float(w["mv2_init"]), scale2[:-1]]) + 1e-6
    shock = np.minimum(mv / prev_scale, 50.0)
    big = shock > 6.0
    since_big = idx - np.maximum.accumulate(np.where(big, idx, -1))   # counts from the sequence start if none yet
    g3 = np.column_stack([mv_s, upd_s, tr_s, np.log1p(mv_s / (tr_s + 0.01)),
                          np.log((_ema(mv, 0.1) + 1e-3) / (mv_s + 1e-3)), np.log((_ema(uf, 0.1) + 1e-3) / (upd_s + 1e-3)),
                          np.log((_ema(tf, 0.1) + 1e-3) / (tr_s + 1e-3)), shock, np.log1p(since_big), _rolling_count(big.astype(np.float64), 1000)])

    # 4. reversal + outside signal
    last_idx = np.maximum.accumulate(np.where(upd, idx, 0))
    last_move = pstep[last_idx]
    cum = np.cumsum(np.where(upd, pstep, 0.0))
    n_upd = np.cumsum(upd)
    upd_pos = np.flatnonzero(upd)                                     # step index of the k-th update
    k6 = n_upd - 6
    base = np.where(k6 >= 0, cum[upd_pos[np.maximum(k6, 0)]], 0.0)
    sum5 = cum - base
    a = x[:, 104:112]
    fam = (a[:, 0] + a[:, 4] - a[:, 7]) / 3.0
    da6 = np.zeros(L)
    da6[1:] = np.diff(a[:, 6])
    g4 = np.column_stack([last_move, sum5, last_move[:, None] * net, _ema(fam, 0.05), _ema(fam, 0.01), _ema(a[:, 1], 0.1), da6])
    return np.column_stack([g1, g2, g3, g4])


N_MOMENTUM = 36
MOM_CLIP = 20.0


def _bc_ema(v: np.ndarray, alpha: float) -> np.ndarray:
    """EMA started from zero and bias-corrected by its running weight 1 - (1 - alpha)^(t+1)."""
    e = lfilter([alpha], [1.0, -(1.0 - alpha)], v, axis=0)
    w = -np.expm1(np.log1p(-alpha) * (np.arange(v.shape[0]) + 1.0))
    return e / w.reshape((-1,) + (1,) * (v.ndim - 1))


def _momentum(x: np.ndarray) -> np.ndarray:
    """Pressure momentum for P = (B_t0, B_t1, S_t0, S_t1, net_t0, net_t1); 6 features each, grouped by feature:
    z_fast, z_slow  (P - EMA_0.1 / EMA_0.01 of P) / scale      -- surge size relative to normal; z_fast - z_slow = acceleration
    trade_w         trade-size-weighted EMA of dP / scale        -- pressure moving while trades print
    m1, m5          (P - P at the last / 5th-last price update) / scale  -- momentum on the price-update clock
    persist         (#steps dP > 0 - #steps dP < 0) over the last 20 steps / 20
    scale = sqrt(bias-corrected EMA_0.01 of dP^2) + 1e-3, so every feature except persist is scale-free.
    """
    w = _pressure_weights()
    L = x.shape[0]
    idx = np.arange(L)
    Vz = (x[:, 22:44] - w["vmean"]) / w["vstd"]
    P4 = np.column_stack([Vz[:, :11] @ w["wB"].T, Vz[:, 11:] @ w["wS"].T])
    P = np.column_stack([P4, P4[:, :2] - P4[:, 2:]])
    dP = np.zeros_like(P)
    dP[1:] = np.diff(P, axis=0)
    scale = np.sqrt(_bc_ema(dP ** 2, 0.01)) + 1e-3
    z_fast = (P - _ema(P, 0.1)) / scale
    z_slow = (P - _ema(P, 0.01)) / scale
    tsize = (x[:, 48:52].max(axis=1) - x[:, 48:52].min(axis=1))[:, None]
    trade_w = _bc_ema(dP * tsize, 0.1) / (_bc_ema(tsize, 0.1) + 1e-3) / scale
    upd = np.ones(L, dtype=bool)
    upd[1:] = np.any(x[1:, 0:22] != x[:-1, 0:22], axis=1)
    upd_pos = np.flatnonzero(upd)
    n_upd = np.cumsum(upd)                                            # >= 1 (row 0 counts as an update)
    m1 = (P - P[upd_pos[n_upd - 1]]) / scale
    m5 = (P - P[np.where(n_upd >= 5, upd_pos[np.maximum(n_upd - 5, 0)], 0)]) / scale
    c = np.cumsum(np.sign(dP), axis=0)
    persist = c.copy()
    persist[20:] -= c[:-20]
    persist /= 20.0
    out = np.hstack([z_fast, z_slow, trade_w, m1, m5]).clip(-MOM_CLIP, MOM_CLIP)
    return np.hstack([out, persist])


def build_features(raw: np.ndarray, feature_set: str = "v144") -> np.ndarray:
    """(N, 112) raw -> (N, D) float32 features for the named set."""
    if feature_set == "v144":
        return extract_32_features_sequence(raw)
    if feature_set == "raw":
        return np.ascontiguousarray(raw, dtype=np.float32)
    x = raw.astype(np.float64)
    if feature_set in ("raw_ev_ema", "slim", "pressure", "momentum"):
        base = raw.astype(np.float32)
    elif feature_set == "v144_ev_ema":
        base = extract_32_features_sequence(raw)
    else:
        raise ValueError(f"unknown feature_set {feature_set!r}")
    extra = {"slim": 3, "pressure": 3 + N_PRESSURE, "momentum": N_MOMENTUM}.get(feature_set, 0)
    nb = base.shape[1]
    out = np.empty((x.shape[0], nb + N_EV_EMA + extra), dtype=np.float32)
    out[:, :nb] = base
    _fill_ev_ema(x, out[:, nb:nb + N_EV_EMA])
    if feature_set == "momentum":
        out[:, nb + N_EV_EMA:] = _momentum(x)
    elif extra:
        c = nb + N_EV_EMA
        out[:, c:c + 3] = _trade_keep(x)
        if feature_set == "pressure":
            out[:, c + 3:] = _pressure(x)
    return out


def feature_dim(feature_set: str) -> int:
    return build_features(np.random.default_rng(0).standard_normal((4, 112)).astype(np.float32), feature_set).shape[1]
