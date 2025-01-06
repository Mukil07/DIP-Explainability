#!/bin/bash

#SBATCH -A mobility_arfs
#SBATCH -c 36
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=4-00:00:00
#SBATCH --output=output_DIPX/MAE_DIPX_combined2.txt
#SBATCH --nodelist=gnode105
#SBATCH --partition=ihub


PORT=$((RANDOM % 55 + 12345))
while ss -tuln | grep -q ":$PORT"; do
  PORT=$((RANDOM % 55 + 12345))
done
echo "Free port found: $PORT"

source activate sf
module load u18/cuda/11.7

cd /scratch/mukil/cemformer

TECH=combined_lr2
MODEL=multimae
DATASET=dipx

best=best_${MODEL}_${DATASET}_${TECH}_dir
runs=runs_${MODEL}_${DATASET}_${TECH}

rm -rf $best
rm -rf $runs

python multigpu_v3.py --model $MODEL --batch 1 --num_classes 7 --dataset $DATASET  --port $PORT \
    --technique $TECH  --dropout 0.65 --learning_rate 0.0001 --n_attributes 32 -distributed -combined_bottleneck -bottleneck 