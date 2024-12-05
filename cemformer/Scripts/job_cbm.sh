#!/bin/bash

#SBATCH -A mobility_arfs
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
#SBATCH --output=output_DIPX/I3D_DIPX_EGO_BOTTLENECK_test2.txt
#SBATCH --nodelist=gnode107
#SBATCH --partition=ihub


source activate sf
module load u18/cuda/11.7

cd /scratch/mukil/final/cemformer
# for debug use this 
#python main.py --model memvit --debug debug 
python main_i3d_dipx_cbm.py --model cbm --batch 1 --num_classes 7 --dataset dipx --technique test2

#torchrun --nproc_per_node=1 --master_addr localhost --master_port 16784 main_dipx.py --model memvit_dipx --num_classes 7 --mem_per_layer 2 --batch 1 
#python main_i3d_dipx.py --model i3d --batch 1 --num_classes 7 --dataset dipx

# CUDA_VISBILE_DEVICES=0,1,2 torchrun --nproc_per_node=3 main_dipx.py --model memvit_dipx --num_classes 7 --mem_per_layer 4
# CUDA_VISBILE_DEVICES=0,1,2 torchrun --nproc_per_node=3 main_dipx.py --model memvit_dipx --num_classes 7 --mem_per_layer 6
# CUDA_VISBILE_DEVICES=0,1,2 torchrun --nproc_per_node=3 main_dipx.py --model memvit_dipx --num_classes 7 --mem_per_layer 8
