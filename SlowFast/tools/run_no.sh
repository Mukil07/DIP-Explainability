#!/bin/bash

#SBATCH -A kcis
#SBATCH -c 14
#SBATCH --gres=gpu:1
#SBATCH --time=4-00:00:00
#SBATCH --output=nobottle.txt
#SBATCH --nodelist=gnode120
#SBATCH --partition=lovelace
#SBATCH --qos=kl4




conda activate eye
module load u18/cuda/11.6
cd /scratch/mukil/SlowFast

#no bottleneck 
export PYTHONPATH="${PYTHONPATH}:/scratch/mukil/SlowFast"
python tools/run_net_final.py \
  --cfg configs/Kinetics/MVITv2_S_CBM.yaml \
  --opts TRAIN.BATCH_SIZE 8 TEST.BATCH_SIZE 8 \
  CBM.N_ATTR 0 CBM.MUL_CLASSES 0 \
  CBM.MULTITASK False CBM.BOTTLENECK False \
  CBM.GAZE_CBM False CBM.EGO_CBM False CBM.COMB_BOTTLE False SOLVER.MAX_EPOCH 50 OUTPUT_DIR ./nobottle

