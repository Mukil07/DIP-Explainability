#!/bin/bash

#SBATCH -A mobility_arfs
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=4-00:00:00
#SBATCH --output=cam.txt
#SBATCH --nodelist=gnode105
#SBATCH --partition=ihub

source activate eye
module load u18/cuda/11.6
cd /scratch/mukil/SlowFast

export PYTHONPATH="${PYTHONPATH}:/scratch/mukil/SlowFast"

python tools/eval_net_final.py \
  --cfg configs/Kinetics/MVITv2_S_CBM_eval.yaml \
  --opts TEST.BATCH_SIZE 1 TEST.CHECKPOINT_FILE_PATH weights/checkpoint_epoch_00070.pyth \
  CBM.N_ATTR 0 CBM.MUL_CLASSES 0 \
  CBM.MULTITASK False CBM.BOTTLENECK False \
  CBM.GAZE_CBM False CBM.EGO_CBM False CBM.COMB_BOTTLE False \
  TRAIN.ENABLE True SOLVER.MAX_EPOCH 1 OUTPUT_DIR ./mvit_eval_new