#!/bin/bash

#SBATCH -A mobility_arfs
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=4-00:00:00
#SBATCH --output=nobottle.txt
#SBATCH --nodelist=gnode107
#SBATCH --partition=ihub




conda activate eye
module load u18/cuda/11.6


cd /scratch/mukil/SlowFast

python tools/run_net.py \
  --cfg configs/Kinetics/MVITv2_S_CBM_nobottle.yaml \
  TRAIN.BATCH_SIZE 1 

