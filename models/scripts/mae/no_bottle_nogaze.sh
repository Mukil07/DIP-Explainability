#!/bin/bash

#SBATCH -A kcis
#SBATCH -c 14
#SBATCH --gres=gpu:1
#SBATCH --time=4-00:00:00
#SBATCH --output=output_DIPX/fine_nobottle_2.txt
#SBATCH --nodelist=gnode121
#SBATCH --partition=lovelace
#SBATCH --qos=kl4




source activate sf
module load u18/cuda/11.7

cd /scratch/mukil/models

TECH=nobottle_2
MODEL=multimae_fine
DATASET=dipx

best=best_${MODEL}_${DATASET}_${TECH}_dir
runs=runs_${MODEL}_${DATASET}_${TECH}

# rm -rf $best
# rm -rf $runs
export PYTHONPATH="${PYTHONPATH}:/scratch/mukil/models"
python tools/videomae/mae_dipx.py --model $MODEL --batch 6 --num_classes 7 --dataset $DATASET  \
    --technique $TECH  --learning_rate 0.00005 --n_attributes 0 --ckp /scratch/mukil/models/best_multimae_fine_dipx_nobottle_dir/best_multimae_fine_dipx.pth -resume