#!/bin/bash

#SBATCH -A wasilone11
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
#SBATCH --output=output_DIPX/normalize.txt
#SBATCH --nodelist=gnode078
#SBATCH --partition=long


PORT=$((RANDOM % 55 + 12345))
while ss -tuln | grep -q ":$PORT"; do
  PORT=$((RANDOM % 55 + 12345))
done
echo "Free port found: $PORT"

source activate sf
module load u18/cuda/11.7

cd /scratch/mukil/cemformer



python tools/normalize_data.py --batch 1 
