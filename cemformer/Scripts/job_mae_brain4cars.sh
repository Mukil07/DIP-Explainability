#!/bin/bash

#SBATCH -A mobility_arfs
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=5G
#SBATCH --time=4-00:00:00
#SBATCH --output=output_DIPX/MAE_B4C_02.txt
#SBATCH --nodelist=gnode097
#SBATCH --partition=ihub


source activate sf
module load u18/cuda/11.7

cd /scratch/mukil/cemformer



#python main_multimae.py --model multimae --num_classes 5 

#working mae brain4cars
# CUDA_LAUNCH_BLOCKING=1 python main_multimae_brain.py --model multimae --batch 2 --num_classes 5 --dataset brain4cars  \
#    --technique brain01 --n_attributes 0 --dropout 0.65

CUDA_LAUNCH_BLOCKING=1 python main_brain_mae.py --model multimae --batch 2 --num_classes 5 --dataset brain4cars  \
--technique brain01 --n_attributes 0 --dropout 0.75