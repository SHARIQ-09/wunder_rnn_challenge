"""Inference solution for Wunder RNN Challenge using ScaledWunderGRU and causal features.

Strictly causal, 1-vCPU compliant, standalone (zero external package dependencies beyond numpy & onnxruntime).
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
import onnxruntime as ort

# Static projection matrix W (112, 14) for 1.3us vectorized feature extraction
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


class PredictionModel:
    """Competition entry point implementing predict(data_point) for online scoring."""

    def __init__(self):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.use_per_session_threads = True
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.add_session_config_entry("session.inter_op.allow_spinning", "0")

        model_path = Path(__file__).resolve().parent / "model.onnx"
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )

        # Inspect model input dimensions
        input_meta = {inp.name: inp.shape for inp in self.session.get_inputs()}
        self.hidden_dim = input_meta.get("hidden_0", [1, 1, 128])[2]
        self.input_dim = input_meta.get("features", [1, 1, 144])[2]

        self.current_seq_ix = None
        self.previous_step = None
        self.hidden_0 = np.zeros((1, 1, self.hidden_dim), dtype=np.float32)
        self.hidden_1 = np.zeros((1, 1, self.hidden_dim), dtype=np.float32)

        # Pre-allocated 144-dim feature buffer and state variables
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
        """Fast 32-feature extraction in ~18us per step using matrix projection and stationary inputs."""
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
            self.hidden_0.fill(0.0)
            self.hidden_1.fill(0.0)
            self._reset_features()

        if self.previous_step is not None and step != self.previous_step + 1:
            raise ValueError("rows must arrive in sequence order")
        self.previous_step = step

        state = np.asarray(data_point.state, dtype=np.float32)
        if state.shape != (112,) or not np.isfinite(state).all():
            raise ValueError("expected 112 finite float32 features")

        # Extract features if model uses 144 engineered features
        if self.input_dim == 144:
            feat = self._extract_32_features(state)
        else:
            feat = state

        # Single-step ONNX forward pass
        prediction, self.hidden_0, self.hidden_1 = self.session.run(
            None,
            {
                "features": feat.reshape(1, 1, self.input_dim),
                "hidden_0": self.hidden_0,
                "hidden_1": self.hidden_1,
            },
        )

        if not data_point.need_prediction:
            return None

        return np.asarray(prediction[0, 0], dtype=np.float32)


if __name__ == "__main__":
    import argparse
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from utils import ScorerStepByStep

    parser = argparse.ArgumentParser()
    parser.add_argument("--validation", required=True)
    args = parser.parse_args()
    print(ScorerStepByStep(args.validation).score(PredictionModel()))
