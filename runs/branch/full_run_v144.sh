#!/bin/bash
# BranchGRU on v144 features, recipe as model M2 (v144 H80, 8 epochs), then pairing test with A and the shipped branch model.
cd /e/wnn_starter_pack/wnn_connectome_starterpack
PY=./wnn_env/Scripts/python.exe
dir=checkpoints_branch80_v144_s31
$PY -u src/train.py --feature_set v144 --norm_stats datasets/norm_stats.npz --branch_gru --hidden_dim 80 --epochs 8 --lr 1e-3 \
   --batch_size 8 --chunk_size 2048 --mse_weight 1.0 --pearson_weight 1.5 --seed 31 --val_sequences 150 --final_full_val \
   --num_workers 0 --save_dir $dir --checkpoint_prefix best.pt --submission_dir $dir/sub --submission_zip $dir/sub.zip > runs/branch/full_v144_s31.log 2>&1
$PY runs/eval_full.py $dir/best.pt >> runs/branch/results.txt 2>&1
$PY -u runs/three_gru/eval_ens.py A=checkpoints_norm_ft/best_norm_ft.pt Br=checkpoints_branch80_s21/best.pt BrV=$dir/best.pt > runs/branch/ens_v144.log 2>&1
grep -E "SWA Ens" runs/branch/full_v144_s31.log | cut -c1-100
tail -1 runs/branch/results.txt
cat runs/branch/ens_v144.log
