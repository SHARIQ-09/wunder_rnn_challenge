#!/bin/bash
# Second-seed replica of runs/feat_screen.sh (same recipe, --seed 7, separate dirs).
# Usage: bash runs/pressure/seed_screen.sh SET [SET ...]
cd /e/wnn_starter_pack/wnn_connectome_starterpack
PY=./wnn_env/Scripts/python.exe
for fs in "$@"; do
  stats=datasets/norm_stats_$fs.npz
  dir=runs/screen/fs_${fs}_s7; mkdir -p $dir
  $PY -u src/train.py --feature_set $fs --norm_stats $stats --train_rg_range 0 2500 --epochs 2 --lr 1e-3 --batch_size 8 \
     --chunk_size 2048 --mse_weight 1.0 --val_sequences 150 --seed 7 --save_dir $dir --checkpoint_prefix best.pt \
     --submission_dir $dir/sub --submission_zip $dir/sub.zip > $dir/train.log 2>&1
  $PY runs/eval_full.py $dir/best.pt >> runs/screen/results.txt 2>&1
done
