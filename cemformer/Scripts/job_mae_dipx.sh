#!/bin/bash

#SBATCH -A mobility_arfs
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=5G
#SBATCH --time=4-00:00:00
#SBATCH --output=output_DIPX/MAE_DIPX_03.txt
#SBATCH --nodelist=gnode101
#SBATCH --partition=ihub


source activate sf
module load u18/cuda/11.7

cd /scratch/mukil/cemformer



#python main_multimae.py --model multimae --num_classes 5 

CUDA_LAUNCH_BLOCKING=1 python main_dipx_mae.py --model multimae --batch 1 --num_classes 7 --dataset dipx  \
   --technique combined_bottleneck_batchsize1 --n_attributes 32 --dropout 0.5 -combined_bottleneck -bottleneck --debug debug

# for plotting gradcam 
CUDA_LAUNCH_BLOCKING=1 python plot_gradcam.py --model multimae --batch 1 --num_classes 7 --dataset dipx  \
   --technique gradcam --dropout 0.5 --debug debug --n_attributes 0