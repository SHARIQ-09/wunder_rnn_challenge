# Preserved Best Submission: Model Soup (WP: 0.6781 / 0.7301)

## Metrics Summary
- **100 Validation Sequences (2,000,000 timesteps)**: **0.6781 WP** ($t_0$: 0.6830, $t_1$: 0.6731)
- **Combined 5 Seqs (100,000 timesteps)**: **0.7301 WP** ($t_0$: 0.7068, $t_1$: 0.7533)
- **Sequence 0 (20,000 timesteps)**: **0.7297 WP** ($t_0$: 0.7190, $t_1$: 0.7405)
- **Prior Best Platform Score (Log 832)**: 0.64322 WP
- **Competition Official Baseline**: 0.617052 WP

## Package Details
- **ZIP File**: `submission_0.6781_modelsoup.zip` (0.83 MB, well under 20 MB limit)
- **ONNX Model**: `model.onnx` (0.90 MB)
- **Inference Latency**: ~71.8 us per step on 1 vCPU (~44.9 minutes for 37.5M test rows, comfortably under 60-minute limit)
- **Checkpoint Source**: `checkpoints/modelsoup_best_0.6781.pt`
- **Features Used**: All 144 features (112 raw + 32 engineered microstructure & multi-scale temporal features)
- **Architecture**:
  - `input_proj`: `Linear(144, 128) -> LayerNorm(128)`
  - `gru_layers`: 2-layer residual GRU ($H=128$, `dropout=0.05`)
  - `head`: `LayerNorm(128) -> Linear(128, 128) -> GELU() -> Dropout -> Linear(128, 2)`
  - Zero-latency multi-seed Model Soup ($0.5 \times W_{\text{seed 42}} + 0.5 \times W_{\text{seed 123}}$)
