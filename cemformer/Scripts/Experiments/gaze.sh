#!/bin/bash

#SBATCH -A mukilan
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
#SBATCH --output=output_DIPX/MAE_DIPX_gaze.txt
#SBATCH --nodelist=gnode059
#SBATCH --partition=long




source activate sf
module load u18/cuda/11.7

cd /scratch/mukil/cemformer

TECH=gaze
MODEL=multimae
DATASET=dipx

best=best_${MODEL}_${DATASET}_${TECH}_dir
runs=runs_${MODEL}_${DATASET}_${TECH}

rm -rf $best
rm -rf $runs

python final_train_single.py --model $MODEL --batch 1 --num_classes 7 --dataset $DATASET  \
    --technique $TECH  --dropout 0.65 --accumulation 4 --learning_rate 0.0001 --n_attributes 15 --multitask_classes 17 -gaze_cbm -bottleneck -multitask