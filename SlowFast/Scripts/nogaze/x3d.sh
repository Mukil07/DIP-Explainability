#!/bin/bash

#SBATCH -A mobility_arfs
#SBATCH -c 11
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=4-00:00:00
#SBATCH --output=output_DIPX/x3d_base.txt
#SBATCH --nodelist=gnode103
#SBATCH --partition=ihub


source activate eye
module load u18/cuda/11.6
cd /scratch/mukil/SlowFast



export PYTHONPATH="${PYTHONPATH}:/scratch/mukil/SlowFast"

python tools/run_net_final.py   --cfg configs/Kinetics/X3D_S.yaml   --opts NUM_GPUS 1   TRAIN.BATCH_SIZE 8 SOLVER.MAX_EPOCH 100