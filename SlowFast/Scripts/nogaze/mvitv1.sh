#!/bin/bash

#SBATCH -A mobility_arfs
#SBATCH -c 11
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=4-00:00:00
#SBATCH --output=output_DIPX/c3d_base.txt
#SBATCH --nodelist=gnode103
#SBATCH --partition=ihub


source activate eye
module load u18/cuda/11.6
cd /scratch/mukil/SlowFast



export PYTHONPATH="${PYTHONPATH}:/scratch/mukil/SlowFast"

 #python tools/run_net_final.py   --cfg configs/Kinetics/C2D_8x8_R50.yaml   --opts NUM_GPUS 1   TRAIN.BATCH_SIZE 8 SOLVER.MAX_EPOCH 100

python tools/run_net_mvit.py \
    --cfg configs/Kinetics/MVIT_B_16x4_CONV_CBM.yaml \
    --opts TRAIN.BATCH_SIZE 8 TEST.BATCH_SIZE 8 \
    TRAIN.AUTO_RESUME False SOLVER.MAX_EPOCH 200 MVIT.LATE_AVG False  OUTPUT_DIR ./mvitv1