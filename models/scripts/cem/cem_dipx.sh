#!/bin/bash

#SBATCH -A mukilan
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
#SBATCH --output=output_DIPX/ego_cem_final2.txt
#SBATCH --nodelist=gnode067
#SBATCH --partition=long




source activate sf
module load u18/cuda/11.7

cd /scratch/mukil/models

TECH=ego_cem_final_2
MODEL=memvit_dipx
DATASET=dipx

best=best_${MODEL}_${DATASET}_${TECH}_dir
runs=runs_${MODEL}_${DATASET}_${TECH}

rm -rf $best
rm -rf $runs

# python aria_gaze.py --model $MODEL --batch 1 --num_classes 7 --dataset $DATASET  \
#     --technique $TECH  --dropout 0.65 --accumulation 4 --learning_rate 0.0001 --n_attributes 17 --multitask_classes 15 -ego_cbm -bottleneck -multitask
export PYTHONPATH="${PYTHONPATH}:/scratch/mukil/models"
python tools/cemformer/main_dipx_cem.py --model $MODEL --batch 1 --num_classes 7 --dataset $DATASET      --technique $TECH  --dropout 0.4 --accumulation 4\
  --learning_rate 0.0001 --n_attributes 17 --multitask_classes 15 -ego_cbm -bottleneck -multitask