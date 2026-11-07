#!/bin/bash
# BranchClsGRU full run with the branch recipe, then pairing test and per-unit scores. Bar: A+Cl vs A+Br 0.685330.
cd /e/wnn_starter_pack/wnn_connectome_starterpack
PY=./wnn_env/Scripts/python.exe
dir=checkpoints_cls80_s41
A=checkpoints_norm_ft/best_norm_ft.pt; Br=checkpoints_branch80_s21/best.pt
$PY -u src/train.py --feature_set raw_ev_ema --norm_stats datasets/norm_stats_raw_ev_ema.npz --branch_gru --cls_head --hidden_dim 80 --epochs 8 --lr 1e-3 \
   --batch_size 8 --chunk_size 2048 --mse_weight 1.0 --pearson_weight 1.5 --seed 41 --val_sequences 150 --final_full_val \
   --num_workers 0 --save_dir $dir --checkpoint_prefix best.pt --submission_dir $dir/sub --submission_zip $dir/sub.zip > runs/cls/full_s41.log 2>&1
$PY runs/eval_full.py $dir/best.pt >> runs/cls/results.txt 2>&1
$PY -u runs/three_gru/eval_ens.py A=$A Br=$Br Cl=$dir/best.pt > runs/cls/ens.log 2>&1
$PY -u runs/cls/eval_units.py Br=$Br Cl=$dir/best.pt A+Br=$A,$Br A+Cl=$A,$dir/best.pt > runs/cls/units.log 2>&1
grep -E "SWA Ens" runs/cls/full_s41.log | cut -c1-100
tail -1 runs/cls/results.txt
cat runs/cls/ens.log runs/cls/units.log
