#!/bin/bash

#SBATCH -A mobility_arfs
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=4-00:00:00
#SBATCH --output=output_DIPX/CEM_BRAIN4CARS_02s.txt
#SBATCH --nodelist=gnode101
#SBATCH --partition=ihub


source activate sf
module load u18/cuda/11.7

cd /scratch/mukil/final/models
# for debug use this 
#python main.py --model memvit --debug debug 

## GUIDELINES 
# for gaze bottleneck - set n_attributes=15, set multitask_classes=17, set -gaze_cbm flag
# for ego bottleneck - set n_attributers=17, set multitask_classes=15, set -ego_cbm flag
# for multitask setup - set n_attributes=0, set -multitask flag
# for no bottleneck setup - set n_attibutes=0


python tools/cemformer/main_brain4cars_cem.py --model memvit --batch 1 --num_classes 5 \
    --dataset brain4cars --technique reduced_size --n_attributes 0 --dropout 0 --mem_per_layer 4