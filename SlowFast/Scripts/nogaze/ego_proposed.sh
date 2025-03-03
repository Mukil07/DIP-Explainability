#!/bin/bash

#SBATCH -A kcis
#SBATCH -c 14
#SBATCH --gres=gpu:1
#SBATCH --time=4-00:00:00
#SBATCH --output=output_DIPX/ego_proposed_5.txt
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
  CBM.N_ATTR 17 CBM.MUL_CLASSES 15 \
  CBM.MULTITASK True CBM.BOTTLENECK True \
  CBM.GAZE_CBM False CBM.EGO_CBM True CBM.COMB_BOTTLE False CBM.CLUSTER 5 TRAIN.AUTO_RESUME False SOLVER.MAX_EPOCH 200 MVIT.LATE_AVG True  OUTPUT_DIR ./egoproposed_5

