#!/bin/bash

#SBATCH -A wasilone11
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
#SBATCH --output=output_DIPX/I3D_DIPX_EGO_BOTTLENECK.txt
#SBATCH --nodelist=gnode078
#SBATCH --partition=long


source activate sf
module load u18/cuda/11.7

cd /scratch/mukil/cemformer
# for debug use this 
#python main.py --model memvit --debug debug 

## GUIDELINES 
# for gaze bottleneck - set n_attributes=15, set multitask_classes=17, set -gaze_cbm flag
# for ego bottleneck - set n_attributers=17, set multitask_classes=15, set -ego_cbm flag
# for multitask setup - set n_attributes=0, set -multitask flag
# for no bottleneck setup - set n_attibutes=0


TECH=ego
MODEL=cbm
DATASET=dipx

python plot_gradcam.py --model $MODEL --batch 1 --num_classes 7 --dataset $DATASET --technique $TECH \
    --n_attributes 17 --multitask_classes 15 --dropout 0.45 -ego_cbm -multitask -bottleneck
