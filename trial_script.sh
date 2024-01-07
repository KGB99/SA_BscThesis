#!/bin/bash

#SBATCH -n 1
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=20G
#SBATCH --time=4:00:00
#module load gcc/8.2.0 python_gpu/3.11.2 eth_proxy
source myenv/bin/activate
python3 cluster_trial.py
deactivate
