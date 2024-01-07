#!/bin/bash

#SBATCH -n 1
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=80G
#SBATCH --time=4:00:00
source myenv/bin/activate
python3 cluster_trial.py --labels_file=/cluster/project/infk/cvg/heinj/students/kbirgi/Annotations/trainSSD/amodal_labels.json --images_dir=/cluster/project/infk/cvg/heinj/datasets/bop/mvpsp
deactivate
