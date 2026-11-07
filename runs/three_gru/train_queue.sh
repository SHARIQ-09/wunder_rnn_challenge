#!/bin/bash
# Three hidden-80 GRUs for a 3-model C-kernel ensemble (fits the time budget like the current 2 x 128 pair).
# Same full recipe as checkpoints_raw_ev_ema_s21 (8 epochs, full-valid selection), one model at a time.
cd /e/wnn_starter_pack/wnn_connectome_starterpack
PY=./wnn_env/Scripts/python.exe
run() {  # name feature_set norm_stats seed
  dir=checkpoints_h80_$1
  $PY -u src/train.py --feature_set $2 --norm_stats $3 --hidden_dim 80 --epochs 8 --lr 1e-3 --batch_size 8 --chunk_size 2048 \
     --mse_weight 1.0 --pearson_weight 1.5 --seed $4 --val_sequences 150 --final_full_val --num_workers 0 \
     --save_dir $dir --checkpoint_prefix best.pt --submission_dir $dir/sub --submission_zip $dir/sub.zip > runs/three_gru/$1.log 2>&1
  $PY runs/eval_full.py $dir/best.pt >> runs/three_gru/results.txt 2>&1
}
run ev_s101 raw_ev_ema datasets/norm_stats_raw_ev_ema.npz 101
run v144_s102 v144 datasets/norm_stats.npz 102
run ev_s103 raw_ev_ema datasets/norm_stats_raw_ev_ema.npz 103
