#!/bin/bash
# Second BranchGRU seed, same recipe as full_run.sh, then pairing test with A and the first branch model.
cd /e/wnn_starter_pack/wnn_connectome_starterpack
PY=./wnn_env/Scripts/python.exe
dir=checkpoints_branch80_s22
$PY -u src/train.py --feature_set raw_ev_ema --norm_stats datasets/norm_stats_raw_ev_ema.npz --branch_gru --hidden_dim 80 --epochs 8 --lr 1e-3 \
   --batch_size 8 --chunk_size 2048 --mse_weight 1.0 --pearson_weight 1.5 --seed 22 --val_sequences 150 --final_full_val \
   --num_workers 0 --save_dir $dir --checkpoint_prefix best.pt --submission_dir $dir/sub --submission_zip $dir/sub.zip > runs/branch/full_s22.log 2>&1
$PY runs/eval_full.py $dir/best.pt >> runs/branch/results.txt 2>&1
$PY -u runs/three_gru/eval_ens.py A=checkpoints_norm_ft/best_norm_ft.pt Br=checkpoints_branch80_s21/best.pt Br2=$dir/best.pt > runs/branch/ens_s22.log 2>&1
grep -E "SWA Ens" runs/branch/full_s22.log | cut -c1-100
tail -1 runs/branch/results.txt
cat runs/branch/ens_s22.log
