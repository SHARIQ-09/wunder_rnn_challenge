#!/bin/bash
# Full run of the momentum set with model B's recipe (checkpoints_raw_ev_ema_s21), then full-valid score.
cd /e/wnn_starter_pack/wnn_connectome_starterpack
PY=./wnn_env/Scripts/python.exe
dir=checkpoints_momentum_s21
$PY -u src/train.py --feature_set momentum --norm_stats datasets/norm_stats_momentum.npz --hidden_dim 128 --epochs 8 --lr 1e-3 \
   --batch_size 8 --chunk_size 2048 --mse_weight 1.0 --pearson_weight 1.5 --seed 21 --val_sequences 150 --final_full_val \
   --num_workers 0 --save_dir $dir --checkpoint_prefix best.pt --submission_dir $dir/sub --submission_zip $dir/sub.zip > runs/momentum/full_s21.log 2>&1
$PY runs/eval_full.py $dir/best.pt >> runs/momentum/results.txt 2>&1
$PY -u runs/three_gru/eval_ens.py A=checkpoints_norm_ft/best_norm_ft.pt B=checkpoints_raw_ev_ema_s21/best.pt Mo=$dir/best.pt > runs/momentum/ens.log 2>&1
