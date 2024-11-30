#!/bin/bash

#SBATCH -A mobility_arfs
#SBATCH -c 16
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
#SBATCH --output=output_DIPX/CEM_Brain.txt
#SBATCH --nodelist=gnode108
#SBATCH --partition=ihub

source activate sf
module load u18/cuda/11.7

cd /scratch/mukil/final/cemformer
# for debug use this 
#python main.py --model memvit --debug debug 
# python main.py --model memvit --mem_per_layer 2
# python main.py --model memvit --mem_per_layer 4
# python main.py --model memvit --mem_per_layer 8
# python main.py --model memvit --mem_per_layer 10
# python main.py --model memvit --mem_per_layer 12

OMP_NUM_THREADS=1
python main_test.py --model memvit --num_classes 5 --mem_per_layer 2 --batch 2
# torchrun --nproc_per_node=1 --master_addr localhost --master_port 12357 main.py --model memvit --num_classes 5 --mem_per_layer 4 --batch 2
# torchrun --nproc_per_node=1 --master_addr localhost --master_port 12358 main.py --model memvit --num_classes 5 --mem_per_layer 6 --batch 2
# torchrun --nproc_per_node=1 --master_addr localhost --master_port 12359 main.py --model memvit --num_classes 5 --mem_per_layer 8 --batch 2
# torchrun --nproc_per_node=1 --master_addr localhost --master_port 12340 main.py --model memvit --num_classes 5 --mem_per_layer 10 --batch 2
# python tools/run_net.py \
#   --cfg configs/masked_ssl/k400_VIT_L_16x4_FT.yaml \
#   TRAIN.CHECKPOINT_FILE_PATH /scratch/mukil/SlowFast/checkpoints/ssl_checkpoint_epoch_00200.pyth