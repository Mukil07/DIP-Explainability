#!/bin/bash

#SBATCH -A wasilone11
#SBATCH -c 36
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
#SBATCH --output=output_DIPX/MAE_DIPX_ego.txt
#SBATCH --nodelist=gnode078
#SBATCH --partition=long


PORT=$((RANDOM % 55 + 12345))
while ss -tuln | grep -q ":$PORT"; do
  PORT=$((RANDOM % 55 + 12345))
done
echo "Free port found: $PORT"

source activate sf
module load u18/cuda/11.7

cd /scratch/mukil/cemformer

TECH=ego_lr
MODEL=multimae
DATASET=dipx

best=best_${MODEL}_${DATASET}_${TECH}_dir
runs=runs_${MODEL}_${DATASET}_${TECH}

rm -rf $best
rm -rf $runs

python multigpu_v2.py --model $MODEL --batch 1 --num_classes 7 --dataset $DATASET  --port $PORT \
    --technique $TECH  --learning_rate 0.0001 --n_attributes 17 --multitask_classes 15 --dropout 0.65 -distributed -ego_cbm -multitask -bottleneck