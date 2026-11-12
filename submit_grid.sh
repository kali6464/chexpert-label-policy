#!/bin/bash
# Run order (once):
#   1. python preprocess_resize.py --src_root <CheXpert-v1.0> --dst_root <CheXpert-v1.0-384> --size 384 --workers 8
#   2. python make_split.py --train_csv <CheXpert-v1.0-384>/train.csv --out_dir <CheXpert-v1.0-384>
#   3. bash submit_grid.sh            # 3 backbones x 2 label policies = 6 jobs
#   4. python compare.py runs/<run_a> runs/<run_b>
#
# Learning rates are sensible starting points for each architecture, not tuned values.
# Add more seeds by extending SEEDS, e.g. SEEDS="42 43 44".

SEEDS="43 44"
POLICIES="zeros Atelectasis"

for SEED in $SEEDS; do
  for POLICY in $POLICIES; do
    for BACKBONE in densenet121 deit_base_distilled_patch16_384 convnext_base.fb_in22k_ft_in1k; do
      case $BACKBONE in
        densenet121)                      LR=1e-4; WD=1e-5 ;;
        deit_base_distilled_patch16_384)  LR=3e-5; WD=0.05 ;;
        convnext_base.fb_in22k_ft_in1k)   LR=5e-5; WD=0.05 ;;
      esac
      NAME="cx_${BACKBONE:0:6}_${POLICY:0:5}_s${SEED}"
      qsub -N "$NAME" -v BACKBONE=$BACKBONE,POLICY=$POLICY,LR=$LR,WD=$WD,SEED=$SEED train.pbs
    done
  done
done
