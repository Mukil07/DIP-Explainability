#!/bin/bash

#SBATCH -A mobility_arfs
#SBATCH -c 20
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=3G
#SBATCH --time=4-00:00:00
#SBATCH --output=output_DIPX/MAE_DIPX_lr2.txt
#SBATCH --nodelist=gnode097
#SBATCH --partition=ihub


PORT=$((RANDOM % 55 + 12345))
while ss -tuln | grep -q ":$PORT"; do
  PORT=$((RANDOM % 55 + 12345))
done
echo "Free port found: $PORT"

source activate sf
module load u18/cuda/11.7

cd /scratch/mukil/cemformer

TECH=lr2
MODEL=multimae
DATASET=dipx

best=best_${MODEL}_${DATASET}_${TECH}_dir
runs=runs_${MODEL}_${DATASET}_${TECH}

rm -rf $best
rm -rf $runs

python multigpu_lr1.py --model $MODEL --batch 1 --num_classes 7 --dataset $DATASET  \
    --weight_decay 0.05 --learning_rate 0.001 --port $PORT \
    --technique $TECH  --dropout 0.65 --n_attributes 0 -distributed 