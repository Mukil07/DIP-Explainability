#!/bin/bash

#SBATCH -A kcis
#SBATCH -c 14
#SBATCH --gres=gpu:1
#SBATCH --time=4-00:00:00
#SBATCH --output=i3d_lstm.txt
#SBATCH --nodelist=gnode120
#SBATCH --partition=lovelace
#SBATCH --job-name=ego_lstm
#SBATCH --qos=kl4

source activate sf
module load u18/cuda/11.7

cd /scratch/mukil/cemformer

TECH=i3d_ego_lstm
DATASET=dipx

best=best_${MODEL}_${DATASET}_${TECH}_dir
runs=runs_${MODEL}_${DATASET}_${TECH}

rm -rf $best
rm -rf $runs

export PYTHONPATH="${PYTHONPATH}:/scratch/mukil/cemformer"


#lstm 
MODEL=i3d_lstm

python i3d/i3d_final.py --model $MODEL --batch 8 --num_classes 7 --dataset $DATASET --technique $TECH \
    --n_attributes 17 --multitask_classes 15 -ego_cbm -multitask -bottleneck

