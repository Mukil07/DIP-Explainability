#!/bin/bash

#SBATCH -A mobility_arfs
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=4-00:00:00
#SBATCH --output=combined.txt
#SBATCH --nodelist=gnode107
#SBATCH --partition=ihub




conda activate eye
module load u18/cuda/11.6
cd /scratch/mukil/SlowFast

export PYTHONPATH="${PYTHONPATH}:/scratch/mukil/SlowFast"
python tools/run_net.py \
  --cfg configs/Kinetics/MVITv2_S_CBM.yaml \
  --opts TRAIN.BATCH_SIZE 2 TEST.BATCH_SIZE 2 \
  CBM.N_ATTR 32 CBM.MUL_CLASSES 0 \
  CBM.MULTITASK False CBM.BOTTLENECK True \
  CBM.GAZE_CBM False CBM.EGO_CBM False CBM.COMB_BOTTLE True
