#!/bin/bash

#SBATCH -A kcis
#SBATCH -c 14
#SBATCH --gres=gpu:1
#SBATCH --time=4-00:00:00
#SBATCH --output=output_DIPX/ego_MSNE_base.txt
#SBATCH --nodelist=gnode122
#SBATCH --partition=lovelace
#SBATCH --qos=kl4


source activate sf
module load u18/cuda/11.7

cd /scratch/mukilv2/cemformer

TECH=i3d_ego_5_MSNE_base
MODEL=cbm
DATASET=dipx

best=best_${MODEL}_${DATASET}_${TECH}_dir
runs=runs_${MODEL}_${DATASET}_${TECH}

rm -rf $best
rm -rf $runs

# python aria_gaze.py --model $MODEL --batch 1 --num_classes 7 --dataset $DATASET  \
#     --technique $TECH  --dropout 0.65 --accumulation 4 --learning_rate 0.0001 --n_attributes 17 --multitask_classes 15 -ego_cbm -bottleneck -multitask
export PYTHONPATH="${PYTHONPATH}:/scratch/mukilv2/cemformer"

python eval_tsne3.py --model $MODEL --batch 1 --num_classes 7 --dataset $DATASET --technique $TECH \
    --n_attributes 17 --multitask_classes 15  -ego_cbm -multitask -bottleneck 
