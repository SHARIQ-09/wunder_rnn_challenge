#!/bin/bash
# Feature-set screening: same recipe as runs/screen.sh (2,500 train seqs, 2 epochs, H128x2, full-valid score),
# but each set gets its own norm stats. Control (v144) = runs/screen/ctrl_h128x2.
# Usage: bash runs/feat_screen.sh SET [SET ...]
cd /e/wnn_starter_pack/wnn_connectome_starterpack
PY=./wnn_env/Scripts/python.exe
for fs in "$@"; do
  stats=datasets/norm_stats_$fs.npz
  [ -f $stats ] || $PY src/compute_norm_stats.py --feature_set $fs --num_sequences 400 --out $stats > runs/screen/norm_$fs.log 2>&1
  dir=runs/screen/fs_$fs; mkdir -p $dir
  $PY -u src/train.py --feature_set $fs --norm_stats $stats --train_rg_range 0 2500 --epochs 2 --lr 1e-3 --batch_size 8 \
     --chunk_size 2048 --mse_weight 1.0 --val_sequences 150 --save_dir $dir --checkpoint_prefix best.pt \
     --submission_dir $dir/sub --submission_zip $dir/sub.zip > $dir/train.log 2>&1
  $PY runs/eval_full.py $dir/best.pt >> runs/screen/results.txt 2>&1
done
