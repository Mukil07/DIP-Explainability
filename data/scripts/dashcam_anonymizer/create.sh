#!/bin/bash

#SBATCH -A mobility_arfs
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --time=4-00:00:00
#SBATCH --nodelist=gnode106
#SBATCH --partition=ihub

source activate blur
module load u18/cuda/11.7

python create_videos.py --config configs/vid_blur.yml


#python downsample.py

