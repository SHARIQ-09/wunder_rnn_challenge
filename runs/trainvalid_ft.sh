#!/bin/bash
# Platform-only test: does adding the (later-period) validation data help on the test period?
# Each ensemble member: 1 epoch on train + all valid, lr 2e-4, resumed from its checkpoint (norm stats travel in it).
cd /e/wnn_starter_pack/wnn_connectome_starterpack
PY=./wnn_env/Scripts/python.exe
COMMON="--extra_train_path datasets/valid.parquet --reset_epoch --epochs 1 --lr 2e-4 --batch_size 8 --chunk_size 2048 \
 --mse_weight 1.0 --pearson_weight 1.5 --val_sequences 50 --seed 31 --checkpoint_prefix best.pt"
$PY -u src/train.py $COMMON --resume checkpoints_norm_ft/best_norm_ft.pt --save_dir checkpoints_tv_normft \
  --submission_dir checkpoints_tv_normft/sub --submission_zip checkpoints_tv_normft/sub.zip > runs/tv_normft.log 2>&1
$PY -u src/train.py $COMMON --feature_set raw_ev_ema --resume checkpoints_raw_ev_ema_s21/best.pt --save_dir checkpoints_tv_ev \
  --submission_dir checkpoints_tv_ev/sub --submission_zip checkpoints_tv_ev/sub.zip > runs/tv_ev.log 2>&1
