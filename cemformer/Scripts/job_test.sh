#!/bin/bash

#SBATCH -A mobility_arfs
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
#SBATCH --output=output_DIPX/I3D_baseline.txt
#SBATCH --nodelist=gnode108
#SBATCH --partition=ihub

source activate sf
module load u18/cuda/11.7

cd /scratch/mukil/final/cemformer
# for debug use this 
# python main_i3d.py --model i3d --batch 1 --debug debug  

OMP_NUM_THREADS=1

python main_i3d.py --model i3d --batch 1 --dataset brain
