#!/bin/bash
# Fine-tune model B on one group only (lr 2e-4, as earlier fine-tunes), then compare with base. Usage: specialist.sh GROUP
cd /e/wnn_starter_pack/wnn_connectome_starterpack
PY=./wnn_env/Scripts/python.exe
g=$1; dir=checkpoints_spec_g$g
$PY -u src/train.py --feature_set raw_ev_ema --resume checkpoints_raw_ev_ema_s21/best.pt --reset_epoch --train_rg_file runs/groups/train_g$g.npy \
  --val_rg_file runs/groups/valid_g$g.npy --epochs 2 --lr 2e-4 --batch_size 8 --chunk_size 2048 --mse_weight 1.0 --pearson_weight 1.5 --seed 31 \
  --val_sequences 150 --final_full_val --num_workers 0 --save_dir $dir --checkpoint_prefix best.pt --submission_dir $dir/sub --submission_zip $dir/sub.zip > runs/groups/spec_g$g.log 2>&1
$PY -u runs/groups/eval_specialist.py $g $dir/best.pt > runs/groups/spec_g$g.eval.log 2>&1
