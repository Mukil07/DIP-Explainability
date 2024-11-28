#!/bin/bash

#SBATCH -A mobility_arfs
#SBATCH -c 16
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
#SBATCH --output=output_DIPX/CEM_DIPX01.txt
#SBATCH --nodelist=gnode105
#SBATCH --partition=ihub


source activate sf
module load u18/cuda/11.7

cd /scratch/mukil/cemformer
# for debug use this 
#python main.py --model memvit --debug debug 


torchrun --nproc_per_node=1 --master_addr localhost --master_port 16784 main_dipx.py --model memvit_dipx --num_classes 7 --mem_per_layer 2 --batch 1 

# CUDA_VISBILE_DEVICES=0,1,2 torchrun --nproc_per_node=3 main_dipx.py --model memvit_dipx --num_classes 7 --mem_per_layer 4
# CUDA_VISBILE_DEVICES=0,1,2 torchrun --nproc_per_node=3 main_dipx.py --model memvit_dipx --num_classes 7 --mem_per_layer 6
# CUDA_VISBILE_DEVICES=0,1,2 torchrun --nproc_per_node=3 main_dipx.py --model memvit_dipx --num_classes 7 --mem_per_layer 8


# for single gpu use this 
# export RANK=0
# export WORLD_SIZE=1
# export MASTER_ADDR=localhost  
# export MASTER_PORT=29500 
# python main_dipx.py --model memvit --num_classes 7 --mem_per_layer 2 --debug debug
#python eval.py --model memvit --num_classes 7 --mem_per_layer 4 

# python tools/run_net.py \
#   --cfg configs/masked_ssl/k400_VIT_L_16x4_FT.yaml \
#   TRAIN.CHECKPOINT_FILE_PATH /scratch/mukil/SlowFast/checkpoints/ssl_checkpoint_epoch_00200.pyth