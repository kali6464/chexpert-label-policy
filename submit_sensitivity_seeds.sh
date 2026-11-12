#!/bin/bash
# Seeds 43 and 44 for the nine sensitivity settings: 3 backbones x 3 variants x 2 seeds = 18 jobs.
# Seed 42 of each variant already exists in ./runs_sens; the matching baselines (main-grid
# settings, seeds 42-44) already exist in ./runs.
#
#   DRY=1 bash submit_sensitivity_seeds.sh   # print the 18 qsub lines only
#   bash submit_sensitivity_seeds.sh         # submit

for SEED in 43 44; do
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
      RUN_NAME="${SHORT}__AE-${VARIANT}__s${SEED}"
      if [ -e "runs_sens/${RUN_NAME}/results.json" ]; then echo "skip (already finished): ${RUN_NAME}"; continue; fi
      CMD="qsub -N s${SEED}_${BACKBONE:0:5}_${VARIANT} -v BACKBONE=$BACKBONE,LR=$LR,WD=$WD,EPOCHS=$EPOCHS,SEED=$SEED,RUN_NAME=$RUN_NAME train_sens2.pbs"
      if [ -n "$DRY" ]; then echo "$CMD"; else $CMD; fi
    done
  done
done
