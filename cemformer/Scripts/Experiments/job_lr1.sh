#!/bin/bash

#SBATCH -A mobility_arfs
#SBATCH -c 20
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=3G
#SBATCH --time=4-00:00:00
#SBATCH --output=output_DIPX/MAE_DIPX_lr1.txt
#SBATCH --nodelist=gnode097
#SBATCH --partition=ihub


source activate sf
module load u18/cuda/11.7

cd /scratch/mukil/cemformer



#python main_multimae.py --model multimae --num_classes 5 

# CUDA_LAUNCH_BLOCKING=1 python main_dipx_mae.py --model multimae --batch 1 --num_classes 7 --dataset dipx  \
#    --technique combined_bottleneck_batchsize1 --n_attributes 32 --dropout 0.5 -combined_bottleneck -bottleneck 
python multigpu_lr1.py --model multimae --batch 1 --num_classes 7 --dataset dipx  \
    --weight_decay 0.05 --learning_rate 0.01 \
    --technique lr1  --dropout 0.65 --n_attributes 0 -distributed 