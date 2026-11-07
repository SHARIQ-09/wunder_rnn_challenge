#!/bin/bash
# Screening: each config trains on 2,500 train sequences for 2 epochs (normalized stationary features),
# then the selected checkpoint is scored on FULL validation by runs/eval_full.py.
# Usage: bash runs/screen.sh NAME "extra train.py args" [NAME "args" ...]
cd /e/wnn_starter_pack/wnn_connectome_starterpack
PY=./wnn_env/Scripts/python.exe
COMMON="--norm_stats datasets/norm_stats.npz --train_rg_range 0 2500 --epochs 2 --lr 1e-3 --batch_size 8 \
 --chunk_size 2048 --mse_weight 1.0 --val_sequences 150"
while [ $# -gt 0 ]; do
  name=$1; args=$2; shift 2
  dir=runs/screen/$name; mkdir -p $dir
  $PY -u src/train.py $COMMON $args --save_dir $dir --checkpoint_prefix best.pt \
     --submission_dir $dir/sub --submission_zip $dir/sub.zip > $dir/train.log 2>&1
  $PY runs/eval_full.py $dir/best.pt >> runs/screen/results.txt 2>&1
done
