#!/bin/bash

#SBATCH -A mukilan
#SBATCH -c 2
#SBATCH --gres=gpu:0
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
#SBATCH --output=k400_mvitv2.txt
#SBATCH --nodelist=gnode070

conda activate sf 
module load u18/cuda/11.7

# FOR SLOWFAST on DIP front view 

sshpass -p 'X29y#BnRPiSs2j' rsync -avzzt --info=progress2 -e 'ssh -p 22' /scratch/cvit/v3/ kmukilan@10.4.16.30:/mnt/base/dip-x/version/v4/