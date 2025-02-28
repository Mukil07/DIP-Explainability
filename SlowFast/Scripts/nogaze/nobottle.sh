#!/bin/bash

#SBATCH -A kcis
#SBATCH -c 14
#SBATCH --gres=gpu:1
#SBATCH --time=4-00:00:00
#SBATCH --output=output_DIPX/nobottle_nogaze.txt
#SBATCH --nodelist=gnode122
#SBATCH --partition=lovelace
#SBATCH --qos=kl4



source activate eye
module load u18/cuda/11.6
cd /scratch/mukilv2/SlowFast

#ego bottlenekc
BATCH_SIZE=2
NUM_GPU=4

export PYTHONPATH="${PYTHONPATH}:/scratch/mukilv2/SlowFast"
python tools/run_net_final.py \
  --cfg configs/Kinetics/MVITv2_S_CBM.yaml \
  --opts TRAIN.BATCH_SIZE 8 TEST.BATCH_SIZE 8 \
  CBM.N_ATTR 0 CBM.MUL_CLASSES 0 \
  CBM.MULTITASK False CBM.BOTTLENECK False \
  CBM.GAZE_CBM False CBM.EGO_CBM False CBM.COMB_BOTTLE False TRAIN.AUTO_RESUME False SOLVER.MAX_EPOCH 200 MVIT.LATE_AVG True OUTPUT_DIR ./nobottlenogaze

