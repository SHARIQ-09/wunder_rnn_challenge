#!/bin/bash
# BranchGRU screen: same recipe as runs/feat_screen.sh raw_ev_ema. Usage: screen.sh SEED
cd /e/wnn_starter_pack/wnn_connectome_starterpack
PY=./wnn_env/Scripts/python.exe
seed=$1; dir=runs/screen/fs_branch80_s$seed; mkdir -p $dir
$PY -u src/train.py --feature_set raw_ev_ema --norm_stats datasets/norm_stats_raw_ev_ema.npz --branch_gru --hidden_dim 80 \
   --train_rg_range 0 2500 --epochs 2 --lr 1e-3 --batch_size 8 --chunk_size 2048 --mse_weight 1.0 --val_sequences 150 --seed $seed \
   --num_workers 0 --save_dir $dir --checkpoint_prefix best.pt --submission_dir $dir/sub --submission_zip $dir/sub.zip > $dir/train.log 2>&1
$PY runs/eval_full.py $dir/best.pt >> runs/screen/results.txt 2>&1
