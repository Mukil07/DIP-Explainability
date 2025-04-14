#!/bin/bash

#SBATCH -A mukilan
#SBATCH -c 7
#SBATCH --gres=gpu:0
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
#SBATCH --output=down_256.txt
#SBATCH --nodelist=gnode070
#SBATCH --partition=long

source activate sf 
module load u18/cuda/11.7

cd /scratch/cvit/v3/

python downsize.py