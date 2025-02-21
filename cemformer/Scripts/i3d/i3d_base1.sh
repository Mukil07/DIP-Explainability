#!/bin/bash

#SBATCH -A mobility_arfs
#SBATCH -c 11
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=4-00:00:00
#SBATCH --output=output_DIPX/i3d_ego_base1.txt
#SBATCH --nodelist=gnode094
#SBATCH --partition=ihub


source activate sf
module load u18/cuda/11.7

cd /scratch/mukil/cemformer

TECH=i3d_ego_base1
MODEL=i3d_fine
DATASET=dipx

best=best_${MODEL}_${DATASET}_${TECH}_dir
runs=runs_${MODEL}_${DATASET}_${TECH}

rm -rf $best
rm -rf $runs

# python aria_gaze.py --model $MODEL --batch 1 --num_classes 7 --dataset $DATASET  \
#     --technique $TECH  --dropout 0.65 --accumulation 4 --learning_rate 0.0001 --n_attributes 17 --multitask_classes 15 -ego_cbm -bottleneck -multitask
export PYTHONPATH="${PYTHONPATH}:/scratch/mukil/cemformer"

python i3d/i3d_final.py --model $MODEL --batch 8 --num_classes 7 --dataset $DATASET --technique $TECH \
    --n_attributes 17 --multitask_classes 15  -ego_cbm -multitask -bottleneck 

