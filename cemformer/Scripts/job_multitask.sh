#!/bin/bash

#SBATCH -A mobility_arfs
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=5G
#SBATCH --time=4-00:00:00
#SBATCH --output=output_DIPX/I3D_DIPX_MULTITASK_02.txt
#SBATCH --nodelist=gnode096
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



# no bottleneck, but multitask 
python main_i3d_dipx_cbm_gaze.py --model cbm --batch 1 --num_classes 7 --dataset dipx --technique multitask02 \
    --n_attributes 0 --dropout 0.45 -multitask

# no bottleneck, only action classification 

#torchrun --nproc_per_node=1 --master_addr localhost --master_port 16784 main_dipx.py --model memvit_dipx --num_classes 7 --mem_per_layer 2 --batch 1 
#python main_i3d_dipx.py --model i3d --batch 1 --num_classes 7 --dataset dipx

# CUDA_VISBILE_DEVICES=0,1,2 torchrun --nproc_per_node=3 main_dipx.py --model memvit_dipx --num_classes 7 --mem_per_layer 4
# CUDA_VISBILE_DEVICES=0,1,2 torchrun --nproc_per_node=3 main_dipx.py --model memvit_dipx --num_classes 7 --mem_per_layer 6
# CUDA_VISBILE_DEVICES=0,1,2 torchrun --nproc_per_node=3 main_dipx.py --model memvit_dipx --num_classes 7 --mem_per_layer 8
