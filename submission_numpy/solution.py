"""Inference solution for Wunder RNN Challenge using pure NumPy / SciPy GRU.

Strictly causal, 1-vCPU compliant, standalone (zero external package dependencies beyond numpy & scipy).
"""
from __future__ import annotations

import os
from pathlib import Path

# Thread caps as specified by competition rules
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
from scipy.special import expit, erf

# Static projection matrix W (112, 14) for fast vectorized feature extraction
W = np.zeros((112, 14), dtype=np.float32)
W[0:11, 0] = 1.0 / 11.0   # p_bid_0_m
W[11:22, 1] = 1.0 / 11.0  # p_ask_0_m
W[52:63, 2] = 1.0 / 11.0  # p_bid_1_m
W[63:74, 3] = 1.0 / 11.0  # p_ask_1_m
W[22:33, 4] = 1.0         # v_bid_0
W[33:44, 4] = -1.0        # ofi_0
W[74:85, 5] = 1.0         # v_bid_1
W[85:96, 5] = -1.0        # ofi_1
W[44:48, 6] = 1.0         # dp_sum
W[48:52, 7] = 1.0         # dv_sum
W[22, 8] = 1.0; W[33, 8] = -1.0    # L0 imb
W[24, 9] = 1.0; W[35, 9] = -1.0    # L2 imb
W[26, 10] = 1.0; W[37, 10] = -1.0  # L4 imb
W[32, 11] = 1.0; W[43, 11] = -1.0  # L10 imb


class PureNumPyScaledWunderGRU:
    """Ultra-fast pure NumPy / SciPy inference engine for ScaledWunderGRU."""
    def __init__(self, w: dict):
        self.W_in = w["W_in"]
        self.b_in = w["b_in"]
        self.gamma_in = w["gamma_in"]
        self.beta_in = w["beta_in"]
        
        self.W0 = w["W0"]
        self.R0 = w["R0"]
        self.Wb0 = w["Wb0"]
        self.Rb0 = w["Rb0"]
        self.bias_rz0 = self.Wb0[:256] + self.Rb0[:256]
        self.gamma_gru0 = w["gamma_gru0"]
        self.beta_gru0 = w["beta_gru0"]
        
        self.W1 = w["W1"]
        self.R1 = w["R1"]
        self.Wb1 = w["Wb1"]
        self.Rb1 = w["Rb1"]
        self.bias_rz1 = self.Wb1[:256] + self.Rb1[:256]
        self.gamma_gru1 = w["gamma_gru1"]
        self.beta_gru1 = w["beta_gru1"]
        
        self.gamma_head = w["gamma_head"]
        self.beta_head = w["beta_head"]
        self.W_dense = w["W_dense"]
        self.b_dense = w["b_dense"]
        self.W_out = w["W_out"]
        self.b_out = w["b_out"]
        
        # Preallocated buffers
        self.h0 = np.zeros(128, dtype=np.float32)
        self.h1 = np.zeros(128, dtype=np.float32)
        
        self.x_proj = np.zeros(128, dtype=np.float32)
        self.h_in = np.zeros(128, dtype=np.float32)
        
        self.gate_x0 = np.zeros(384, dtype=np.float32)
        self.gate_h0 = np.zeros(384, dtype=np.float32)
        self.rz0 = np.zeros(256, dtype=np.float32)
        self.cand0 = np.zeros(128, dtype=np.float32)
        self.h_gru0 = np.zeros(128, dtype=np.float32)
        
        self.gate_x1 = np.zeros(384, dtype=np.float32)
        self.gate_h1 = np.zeros(384, dtype=np.float32)
        self.rz1 = np.zeros(256, dtype=np.float32)
        self.cand1 = np.zeros(128, dtype=np.float32)
        self.h_gru1 = np.zeros(128, dtype=np.float32)
        
        self.h_head = np.zeros(128, dtype=np.float32)
        self.dense = np.zeros(128, dtype=np.float32)
        self.dense_act = np.zeros(128, dtype=np.float32)
        self.pred = np.zeros(2, dtype=np.float32)

    def reset(self):
        self.h0.fill(0.0)
        self.h1.fill(0.0)

    @staticmethod
    def _layer_norm(x, gamma, beta, eps=1e-5):
        m = x.mean()
        v = x.var()
        return (x - m) / np.sqrt(v + eps) * gamma + beta

    def step(self, x_144: np.ndarray) -> np.ndarray:
        # 1. Input Linear projection + LayerNorm
        np.dot(self.W_in, x_144, out=self.x_proj)
        self.x_proj += self.b_in
        self.h_in = self._layer_norm(self.x_proj, self.gamma_in, self.beta_in)
        
        # 2. GRU 0
        np.dot(self.W0, self.h_in, out=self.gate_x0)
        np.dot(self.R0, self.h0, out=self.gate_h0)
        
        np.add(self.gate_x0[:256], self.gate_h0[:256], out=self.rz0)
        np.add(self.rz0, self.bias_rz0, out=self.rz0)
        expit(self.rz0, out=self.rz0)
        
        r0 = self.rz0[:128]
        z0 = self.rz0[128:]
        
        np.add(self.gate_h0[256:], self.Rb0[256:], out=self.cand0)
        self.cand0 *= r0
        self.cand0 += self.gate_x0[256:]
        self.cand0 += self.Wb0[256:]
        np.tanh(self.cand0, out=self.cand0)
        
        self.h0 *= z0
        self.h0 += (1.0 - z0) * self.cand0
        self.h_gru0 = self._layer_norm(self.h0, self.gamma_gru0, self.beta_gru0)
        
        # 3. GRU 1
        np.dot(self.W1, self.h_gru0, out=self.gate_x1)
        np.dot(self.R1, self.h1, out=self.gate_h1)
        
        np.add(self.gate_x1[:256], self.gate_h1[:256], out=self.rz1)
        np.add(self.rz1, self.bias_rz1, out=self.rz1)
        expit(self.rz1, out=self.rz1)
        
        r1 = self.rz1[:128]
        z1 = self.rz1[128:]
        
        np.add(self.gate_h1[256:], self.Rb1[256:], out=self.cand1)
        self.cand1 *= r1
        self.cand1 += self.gate_x1[256:]
        self.cand1 += self.Wb1[256:]
        np.tanh(self.cand1, out=self.cand1)
        
        self.h1 *= z1
        self.h1 += (1.0 - z1) * self.cand1
        # Inter-layer residual connection
        res = self.h1 + self.h_gru0
        self.h_gru1 = self._layer_norm(res, self.gamma_gru1, self.beta_gru1)
        
        # 4. Head: LayerNorm -> Linear -> GELU -> Linear
        self.h_head = self._layer_norm(self.h_gru1, self.gamma_head, self.beta_head)
        np.dot(self.W_dense, self.h_head, out=self.dense)
        self.dense += self.b_dense
        
        # GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
        self.dense_act = 0.5 * self.dense * (1.0 + erf(self.dense * 0.7071067811865475))
        
        np.dot(self.W_out, self.dense_act, out=self.pred)
        self.pred += self.b_out
        return self.pred


class PredictionModel:
    """Competition entry point implementing predict(data_point) for online scoring."""

    def __init__(self):
        weights_path = Path(__file__).resolve().parent / "weights.npz"
        if not weights_path.exists():
            raise FileNotFoundError(f"weights.npz not found at {weights_path}")
        
        data = np.load(weights_path)
        weights = {k: data[k] for k in data.files}
        self.engine = PureNumPyScaledWunderGRU(weights)

        self.current_seq_ix = None
        self.previous_step = None
        self.W = W
        self.buf_144 = np.zeros(144, dtype=np.float32)
        
        self.prev_mid_0 = None
        self.prev_mid_1 = None
        self.ema_ret_fast = 0.0
        self.ema_ret_med = 0.0
        self.ema_ret_slow = 0.0
        self.ema_ofi_fast = 0.0
        self.ema_ofi_med = 0.0
        self.ema_ofi_slow = 0.0
        self.ema_ret1_med = 0.0
        self.ema_cross_mid = 0.0
        self.ema_var_proxy = 0.0
        self.ema_trade_vol = 0.0
        self.ema_money_flow = 0.0
        self.ema_ret_long = 0.0
        self.ema_ofi1_med = 0.0

    def _reset_features(self):
        self.prev_mid_0 = None
        self.prev_mid_1 = None
        self.ema_ret_fast = 0.0
        self.ema_ret_med = 0.0
        self.ema_ret_slow = 0.0
        self.ema_ofi_fast = 0.0
        self.ema_ofi_med = 0.0
        self.ema_ofi_slow = 0.0
        self.ema_ret1_med = 0.0
        self.ema_cross_mid = 0.0
        self.ema_var_proxy = 0.0
        self.ema_trade_vol = 0.0
        self.ema_money_flow = 0.0
        self.ema_ret_long = 0.0
        self.ema_ofi1_med = 0.0

    def _extract_32_features(self, x: np.ndarray) -> np.ndarray:
        buf = self.buf_144
        lin = x @ self.W

        p_bid_0_m = lin[0]
        p_ask_0_m = lin[1]
        mid_0 = (p_bid_0_m + p_ask_0_m) * 0.5
        p_bid_1_m = lin[2]
        p_ask_1_m = lin[3]
        mid_1 = (p_bid_1_m + p_ask_1_m) * 0.5
        ofi_0 = lin[4]
        ofi_1 = lin[5]
        dp_sum = lin[6]
        dv_sum = lin[7]

        # Raw features stationarized relative to mid price
        buf[:112] = x
        buf[0:22] = x[0:22] - mid_0
        buf[52:74] = x[52:74] - mid_1

        var_b0 = max(float(x[0:11] @ x[0:11]) * (1.0 / 11.0) - p_bid_0_m * p_bid_0_m, 0.0)
        var_a0 = max(float(x[11:22] @ x[11:22]) * (1.0 / 11.0) - p_ask_0_m * p_ask_0_m, 0.0)
        mf = float(x[44:48] @ x[48:52])

        if self.prev_mid_0 is None:
            ret_0 = 0.0
            ret_1 = 0.0
            self.ema_ret_fast = 0.0
            self.ema_ret_med = 0.0
            self.ema_ret_slow = 0.0
            self.ema_ofi_fast = ofi_0
            self.ema_ofi_med = ofi_0
            self.ema_ofi_slow = ofi_0
            self.ema_ret1_med = 0.0
            self.ema_cross_mid = mid_0 - mid_1
            self.ema_var_proxy = 0.0
            self.ema_trade_vol = dv_sum
            self.ema_money_flow = mf
            self.ema_ret_long = 0.0
            self.ema_ofi1_med = ofi_1
        else:
            ret_0 = mid_0 - self.prev_mid_0
            ret_1 = mid_1 - self.prev_mid_1

            self.ema_ret_fast = 0.2 * ret_0 + 0.8 * self.ema_ret_fast
            self.ema_ret_med  = 0.05 * ret_0 + 0.95 * self.ema_ret_med
            self.ema_ret_slow = 0.01 * ret_0 + 0.99 * self.ema_ret_slow

            self.ema_ofi_fast = 0.2 * ofi_0 + 0.8 * self.ema_ofi_fast
            self.ema_ofi_med  = 0.05 * ofi_0 + 0.95 * self.ema_ofi_med
            self.ema_ofi_slow = 0.01 * ofi_0 + 0.99 * self.ema_ofi_slow

            self.ema_ret1_med = 0.05 * ret_1 + 0.95 * self.ema_ret1_med
            self.ema_cross_mid = 0.05 * (mid_0 - mid_1) + 0.95 * self.ema_cross_mid
            self.ema_var_proxy = 0.05 * (ret_0 * ret_0) + 0.95 * self.ema_var_proxy
            self.ema_trade_vol = 0.05 * dv_sum + 0.95 * self.ema_trade_vol
            self.ema_money_flow = 0.05 * mf + 0.95 * self.ema_money_flow
            self.ema_ret_long = 0.002 * ret_0 + 0.998 * self.ema_ret_long
            self.ema_ofi1_med = 0.05 * ofi_1 + 0.95 * self.ema_ofi1_med

        self.prev_mid_0 = mid_0
        self.prev_mid_1 = mid_1

        buf[112] = p_ask_0_m - p_bid_0_m
        buf[113] = np.sqrt(var_b0 + 1e-6)
        buf[114] = np.sqrt(var_a0 + 1e-6)
        buf[115] = p_ask_0_m - p_bid_0_m
        buf[116] = ofi_0
        buf[117] = lin[8]
        buf[118] = lin[9]
        buf[119] = lin[10]
        buf[120] = lin[11]
        buf[121] = p_ask_1_m - p_bid_1_m
        buf[122] = mid_0 - mid_1
        buf[123] = ofi_0 - ofi_1
        buf[124] = dp_sum
        buf[125] = dv_sum
        buf[126] = mf
        buf[127] = ret_0

        buf[128] = self.ema_ret_fast
        buf[129] = self.ema_ret_med
        buf[130] = self.ema_ret_slow
        buf[131] = self.ema_ret_fast - self.ema_ret_med
        buf[132] = self.ema_ofi_fast
        buf[133] = self.ema_ofi_med
        buf[134] = self.ema_ofi_slow
        buf[135] = self.ema_ofi_fast - self.ema_ofi_slow
        buf[136] = self.ema_ret1_med
        buf[137] = self.ema_ret1_med - self.ema_ret_med
        buf[138] = self.ema_cross_mid
        buf[139] = self.ema_var_proxy
        buf[140] = self.ema_trade_vol
        buf[141] = self.ema_money_flow
        buf[142] = self.ema_ret_long
        buf[143] = self.ema_ofi1_med - self.ema_ofi_med

        return buf

    def predict(self, data_point):
        seq_ix = int(data_point.seq_ix)
        step = int(data_point.step_in_seq)

        # Handle sequence reset
        if self.current_seq_ix != seq_ix:
            if step != 0:
                raise ValueError("a new sequence must start at step zero")
            self.current_seq_ix = seq_ix
            self.previous_step = None
            self.engine.reset()
            self._reset_features()

        if self.previous_step is not None and step != self.previous_step + 1:
            raise ValueError("rows must arrive in sequence order")
        self.previous_step = step

        state = np.asarray(data_point.state, dtype=np.float32)
        if state.shape != (112,) or not np.isfinite(state).all():
            raise ValueError("expected 112 finite float32 features")

        feat = self._extract_32_features(state)
        prediction = self.engine.step(feat)

        if not data_point.need_prediction:
            return None

        return np.asarray(prediction, dtype=np.float32)


if __name__ == "__main__":
    import argparse
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from utils import ScorerStepByStep

    parser = argparse.ArgumentParser()
    parser.add_argument("--validation", required=True)
    args = parser.parse_args()
    print(ScorerStepByStep(args.validation).score(PredictionModel()))
