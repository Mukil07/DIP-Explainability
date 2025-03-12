#!/bin/bash

#SBATCH -A kcis
#SBATCH -c 14
#SBATCH --gres=gpu:1
#SBATCH --time=4-00:00:00
#SBATCH --output=output_DIPX/ego_proposed_cammap.txt
#SBATCH --nodelist=gnode122
#SBATCH --partition=lovelace
#SBATCH --qos=kl4


source activate eye
module load u18/cuda/11.6
cd /scratch/mukil/SlowFast

export PYTHONPATH="${PYTHONPATH}:/scratch/mukilv2/SlowFast"

python tools/eval_net_final.py \
  --cfg configs/Kinetics/MVITv2_S_CBM_eval.yaml \
  --opts TEST.BATCH_SIZE 8 TEST.CHECKPOINT_FILE_PATH /scratch/mukilv2/SlowFast/weight/checkpoint_epoch_00190.pyth \
  CBM.N_ATTR 17 CBM.MUL_CLASSES 15 \
  CBM.MULTITASK True CBM.BOTTLENECK True \
  CBM.GAZE_CBM False CBM.EGO_CBM True CBM.COMB_BOTTLE False MVIT.LATE_AVG True \
  TRAIN.ENABLE True SOLVER.MAX_EPOCH 1 OUTPUT_DIR ./mvit_eval_new_cammaps