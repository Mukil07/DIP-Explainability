#!/bin/bash

#SBATCH -A mobility_arfs
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --time=4-00:00:00
#SBATCH --nodelist=gnode106
#SBATCH --partition=ihub

source activate blur
module load u18/cuda/11.7
cd /scratch/mukil/data/scripts/dashcam_anonymizer

export PYTHONPATH=/scratch/mukil/data/scripts/dashcam_anonymizer

python blur_videos.py --config configs/vid_blur_gaze.yaml 




