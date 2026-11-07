#!/bin/bash
# Fine-tune model B on unit-0.6569 sequences only, same recipe as runs/groups/specialist.sh, then compare with base.
cd /e/wnn_starter_pack/wnn_connectome_starterpack
PY=./wnn_env/Scripts/python.exe
dir=checkpoints_spec_u0
$PY -c "import numpy as np
for s in ('train','valid'): np.save(f'runs/decode/{s}_u0.npy', np.flatnonzero(np.load(f'runs/decode/{s}_unit.npy')==0))"
$PY -u src/train.py --feature_set raw_ev_ema --resume checkpoints_raw_ev_ema_s21/best.pt --reset_epoch --train_rg_file runs/decode/train_u0.npy \
  --val_rg_file runs/decode/valid_u0.npy --epochs 2 --lr 2e-4 --batch_size 8 --chunk_size 2048 --mse_weight 1.0 --pearson_weight 1.5 --seed 31 \
  --val_sequences 150 --final_full_val --num_workers 0 --save_dir $dir --checkpoint_prefix best.pt --submission_dir $dir/sub --submission_zip $dir/sub.zip > runs/decode/spec_u0.log 2>&1
echo train exit $?
$PY -u runs/decode/eval_spec_u0.py $dir/best.pt > runs/decode/spec_u0.eval.log 2>&1
echo eval exit $?
cat runs/decode/spec_u0.eval.log
