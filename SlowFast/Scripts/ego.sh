#!/bin/bash

#SBATCH -A mukilan
#SBATCH -c 36
#SBATCH --gres=gpu:4
#SBATCH --time=4-00:00:00
#SBATCH --output=ego_multi.txt
#SBATCH --nodelist=gnode071
#SBATCH --partition=long



source activate eye
module load u18/cuda/11.6
cd /scratch/mukil/SlowFast

#ego bottlenekc
BATCH_SIZE=2
NUM_GPU=4

export PYTHONPATH="${PYTHONPATH}:/scratch/mukil/SlowFast"
python tools/run_net_final.py \
  --cfg configs/Kinetics/MVITv2_S_CBM.yaml \
  --opts TRAIN.BATCH_SIZE 8 TEST.BATCH_SIZE 8 NUM_GPUS 4 DATA_LOADER.NUM_WORKERS 1 \
  CBM.N_ATTR 17 CBM.MUL_CLASSES 15 \
  CBM.MULTITASK True CBM.BOTTLENECK True \
  CBM.GAZE_CBM False CBM.EGO_CBM True CBM.COMB_BOTTLE False TRAIN.AUTO_RESUME False SOLVER.MAX_EPOCH 200 OUTPUT_DIR ./ego 

