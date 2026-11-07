#!/bin/bash
# Scoring-mask experiment: fine-tune on valid[0:1400], evaluate on held-out valid[1400:1873]
cd /e/wnn_starter_pack/wnn_connectome_starterpack
PY=./wnn_env/Scripts/python.exe
COMMON="--train_path datasets/valid.parquet --train_rg_range 0 1400 --val_rg_range 1400 1873 \
 --resume checkpoints_5epoch/best_model_5ep.pt --reset_epoch --norm_stats datasets/norm_stats.npz \
 --epochs 3 --lr 2e-4 --batch_size 8 --chunk_size 2048 --pearson_weight 1.5 --val_sequences -1 --final_full_val --seed 11"
for arm in plain:none w5:5 scoredonly:inf; do
  name=${arm%%:*}; w=${arm##*:}
  extra=""; [ "$w" != "none" ] && extra="--scored_weight $w"
  $PY -u src/train.py $COMMON $extra --save_dir runs/mask_$name --checkpoint_prefix best.pt \
    --submission_dir runs/mask_$name/sub --submission_zip runs/mask_$name/sub.zip > runs/mask_$name.log 2>&1
done
