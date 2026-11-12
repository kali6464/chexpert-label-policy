#!/bin/bash
# U-Ignore baseline: uncertain labels are left out of the loss.
# 3 backbones x 3 seeds = 9 jobs, same settings as the main grid. Uses the existing train.pbs.
# Results go to ./runs next to the main grid (names like densenet121__ignore__s42).
#
# Run from the CheXpert_pipeline folder, after the smoke test has passed:
#   DRY=1 bash submit_ignore.sh     # print the 9 qsub lines only
#   bash submit_ignore.sh           # submit

for SEED in 42 43 44; do
  for BACKBONE in densenet121 deit_base_distilled_patch16_384 convnext_base.fb_in22k_ft_in1k; do
    case $BACKBONE in
      densenet121)                      LR=1e-4; WD=1e-5 ;;
      deit_base_distilled_patch16_384)  LR=3e-5; WD=0.05 ;;
      convnext_base.fb_in22k_ft_in1k)   LR=5e-5; WD=0.05 ;;
    esac
    RUN="runs/${BACKBONE%%.*}__ignore__s${SEED}"
    if [ -e "$RUN/results.json" ]; then echo "skip (already finished): $RUN"; continue; fi
    CMD="qsub -N ign_${BACKBONE:0:5}_s${SEED} -v BACKBONE=$BACKBONE,POLICY=ignore,LR=$LR,WD=$WD,SEED=$SEED train.pbs"
    if [ -n "$DRY" ]; then echo "$CMD"; else $CMD; fi
  done
done
