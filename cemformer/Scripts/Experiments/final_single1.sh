#!/bin/bash

#SBATCH -A mobility_arfs
#SBATCH -c 9
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=4-00:00:00
#SBATCH --output=output_DIPX/MAE_DIPX_combined_aria_final_1.txt
#SBATCH --nodelist=gnode100
#SBATCH --partition=ihub


source activate sf
module load u18/cuda/11.7

cd /scratch/mukil/cemformer

TECH=combined_aria_final_1
MODEL=multimae
DATASET=dipx

best=best_${MODEL}_${DATASET}_${TECH}_dir
runs=runs_${MODEL}_${DATASET}_${TECH}

rm -rf $best
rm -rf $runs

python final_train_single.py --model $MODEL --batch 1 --num_classes 7 --dataset $DATASET  \
    --technique $TECH  --dropout 0.65 --accumulation 8 --learning_rate 0.0001 --n_attributes 32  -combined_bottleneck -bottleneck 