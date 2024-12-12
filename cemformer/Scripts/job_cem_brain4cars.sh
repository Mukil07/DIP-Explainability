#!/bin/bash

#SBATCH -A mobility_arfs
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=4-00:00:00
#SBATCH --output=output_DIPX/CEM_BRAIN4CARS_01.txt
#SBATCH --nodelist=gnode099
#SBATCH --partition=ihub


source activate sf
module load u18/cuda/11.7

cd /scratch/mukil/final/cemformer
# for debug use this 
#python main.py --model memvit --debug debug 

## GUIDELINES 
# for gaze bottleneck - set n_attributes=15, set multitask_classes=17, set -gaze_cbm flag
# for ego bottleneck - set n_attributers=17, set multitask_classes=15, set -ego_cbm flag
# for multitask setup - set n_attributes=0, set -multitask flag
# for no bottleneck setup - set n_attibutes=0

# python main_i3d_dipx_cbm_gaze.py --model cbm_gaze --batch 1 \
#     --n_attributes 15 --multitask_classes 17 --num_classes 7 --dataset dipx --technique test1only --debug debug -gaze_cbm



# no bottleneck, only action classification 
python main_brain4cars_cem.py --model memvit --batch 1 --num_classes 5 \
    --dataset brain4cars --technique baseline --n_attributes 0 --dropout 0.65 --mem_per_layer 4
