#!/bin/bash

#SBATCH -A kcis
#SBATCH -c 14
#SBATCH --gres=gpu:1
#SBATCH --time=4-00:00:00
#SBATCH --output=output_DIPX/fine_ego.txt
#SBATCH --nodelist=gnode121
#SBATCH --partition=lovelace
#SBATCH --qos=kl4




source activate sf
module load u18/cuda/11.7

cd /scratch/mukil/cemformer

TECH=ego
MODEL=multimae_fine
DATASET=dipx

best=best_${MODEL}_${DATASET}_${TECH}_dir
runs=runs_${MODEL}_${DATASET}_${TECH}

rm -rf $best
rm -rf $runs
export PYTHONPATH="${PYTHONPATH}:/scratch/mukil/cemformer"
python Videomae/mae_dipx.py --model $MODEL --batch 6 --num_classes 7 --dataset $DATASET  \
    --technique $TECH  --learning_rate 0.00005 --n_attributes 17 --multitask_classes 15  -ego_cbm -multitask -bottleneck  