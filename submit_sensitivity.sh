#!/bin/bash
# Backbone sensitivity check: does the "backbones are indistinguishable" result survive a
# different learning rate or a longer schedule?
#
# Policy Atelectasis+Edema, seed 42, 3 backbones x 3 variants = 9 jobs:
#   lr0.5x : half the main-grid learning rate, 3 epochs
#   lr2x   : double the main-grid learning rate, 3 epochs
#   ep6    : main-grid learning rate, 6 epochs
# The matching baseline (main-grid learning rate, 3 epochs, seed 42) already exists in ./runs.
#
# Run from the CheXpert_pipeline folder:   bash submit_sensitivity.sh
# Dry run (prints the qsub lines only):     DRY=1 bash submit_sensitivity.sh

for BACKBONE in densenet121 deit_base_distilled_patch16_384 convnext_base.fb_in22k_ft_in1k; do
  case $BACKBONE in
    densenet121)                      BASE=1e-4; HALF=5e-5;   DOUBLE=2e-4; WD=1e-5 ;;
    deit_base_distilled_patch16_384)  BASE=3e-5; HALF=1.5e-5; DOUBLE=6e-5; WD=0.05 ;;
    convnext_base.fb_in22k_ft_in1k)   BASE=5e-5; HALF=2.5e-5; DOUBLE=1e-4; WD=0.05 ;;
  esac
  SHORT=${BACKBONE%%.*}
  for VARIANT in lr0.5x lr2x ep6; do
    case $VARIANT in
      lr0.5x) LR=$HALF;   EPOCHS=3 ;;
      lr2x)   LR=$DOUBLE; EPOCHS=3 ;;
      ep6)    LR=$BASE;   EPOCHS=6 ;;
    esac
    RUN_NAME="${SHORT}__AE-${VARIANT}__s42"
    if [ -e "runs_sens/${RUN_NAME}/results.json" ]; then
      echo "skip (already finished): ${RUN_NAME}"; continue
    fi
    CMD="qsub -N sens_${BACKBONE:0:5}_${VARIANT} -v BACKBONE=$BACKBONE,LR=$LR,WD=$WD,EPOCHS=$EPOCHS,RUN_NAME=$RUN_NAME train_sens.pbs"
    if [ -n "$DRY" ]; then echo "$CMD"; else $CMD; fi
  done
done
