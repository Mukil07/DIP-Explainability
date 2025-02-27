#!/bin/bash

#SBATCH -A kcis
#SBATCH -c 14
#SBATCH --gres=gpu:1
#SBATCH --time=4-00:00:00
#SBATCH --output=ego_new2.txt
#SBATCH --nodelist=gnode120
#SBATCH --partition=lovelace
#SBATCH --qos=kl4


source activate eye
module load u18/cuda/11.6
cd /scratch/mukil/SlowFast

#ego bottlenekc
export PYTHONPATH="${PYTHONPATH}:/scratch/mukil/SlowFast"
python tools/run_net_final.py \
  --cfg configs/Kinetics/MVITv2_S_CBM.yaml \
  --opts TRAIN.BATCH_SIZE 2 TEST.BATCH_SIZE 2 \
  CBM.N_ATTR 17 CBM.MUL_CLASSES 15 \
  CBM.MULTITASK True CBM.BOTTLENECK True \
  CBM.GAZE_CBM False CBM.EGO_CBM True CBM.COMB_BOTTLE False TRAIN.AUTO_RESUME True SOLVER.MAX_EPOCH 200 OUTPUT_DIR ./ego 
