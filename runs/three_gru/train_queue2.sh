#!/bin/bash
# Continuation of train_queue.sh: wait for model 2 (PID given), score it, run the long-memory screen, then model 3.
cd /e/wnn_starter_pack/wnn_connectome_starterpack
PY=./wnn_env/Scripts/python.exe
while kill -0 $1 2>/dev/null; do sleep 60; done
$PY runs/eval_full.py checkpoints_h80_v144_s102/best.pt >> runs/three_gru/results.txt 2>&1

# Long-memory screen: same recipe as runs/feat_screen.sh raw_ev_ema (control fs_raw_ev_ema 0.668986), + mem_dim 24.
dir=runs/screen/fs_raw_ev_ema_mem24; mkdir -p $dir
$PY -u src/train.py --feature_set raw_ev_ema --norm_stats datasets/norm_stats_raw_ev_ema.npz --train_rg_range 0 2500 --epochs 2 \
   --lr 1e-3 --batch_size 8 --chunk_size 2048 --mse_weight 1.0 --val_sequences 150 --mem_dim 24 --num_workers 0 \
   --save_dir $dir --checkpoint_prefix best.pt --submission_dir $dir/sub --submission_zip $dir/sub.zip > $dir/train.log 2>&1
$PY runs/eval_full.py $dir/best.pt >> runs/screen/results.txt 2>&1

dir=checkpoints_h80_ev_s103
$PY -u src/train.py --feature_set raw_ev_ema --norm_stats datasets/norm_stats_raw_ev_ema.npz --hidden_dim 80 --epochs 8 --lr 1e-3 --batch_size 8 --chunk_size 2048 \
   --mse_weight 1.0 --pearson_weight 1.5 --seed 103 --val_sequences 150 --final_full_val --num_workers 0 \
   --save_dir $dir --checkpoint_prefix best.pt --submission_dir $dir/sub --submission_zip $dir/sub.zip > runs/three_gru/ev_s103.log 2>&1
$PY runs/eval_full.py $dir/best.pt >> runs/three_gru/results.txt 2>&1
