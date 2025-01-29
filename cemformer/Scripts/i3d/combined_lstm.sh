#!/bin/bash

#SBATCH -A mobility_arfs
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=4-00:00:00
#SBATCH --output=output_DIPX/i3d_combined_lstm.txt
#SBATCH --nodelist=gnode110
#SBATCH --partition=ihub




source activate sf
module load u18/cuda/11.7

cd /scratch/mukil/cemformer

TECH=i3d_combined_lstm
DATASET=dipx

best=best_${MODEL}_${DATASET}_${TECH}_dir
runs=runs_${MODEL}_${DATASET}_${TECH}

rm -rf $best
rm -rf $runs

export PYTHONPATH="${PYTHONPATH}:/scratch/mukil/cemformer"


#lstm 
MODEL=i3d_lstm
python i3d/i3d_final.py --model $MODEL --batch 1 --num_classes 7  \
    --dataset $DATASET  --technique $TECH --n_attributes 32 --dropout 0.45 -combined_bottleneck -bottleneck 