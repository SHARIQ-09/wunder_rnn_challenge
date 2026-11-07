#!/bin/bash
# Stage 2: does valid-period fine-tuning stack on top of the normalized train model?
cd /e/wnn_starter_pack/wnn_connectome_starterpack
PY=./wnn_env/Scripts/python.exe
BASE=checkpoints_norm_ft/best_norm_ft.pt
$PY runs/eval_range.py $BASE 1400 1873 > runs/stage2_base_holdout.log 2>&1
$PY -u src/train.py --train_path datasets/valid.parquet --train_rg_range 0 1400 --val_rg_range 1400 1873 \
  --resume $BASE --reset_epoch --epochs 3 --lr 2e-4 --batch_size 8 --chunk_size 2048 --pearson_weight 1.5 \
  --val_sequences -1 --final_full_val --seed 11 --save_dir runs/stage2_holdout --checkpoint_prefix best.pt \
  --submission_dir runs/stage2_holdout/sub --submission_zip runs/stage2_holdout/sub.zip > runs/stage2_holdout.log 2>&1
