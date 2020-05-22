#!/bin/sh

#SBATCH --job-name=full_supervised_aug_cnn
#SBATCH --output=%j_%t_full_supervised_aug_cnn.out
#SBATCH --error=%j_%t_full_supervised_aug_cnn.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --partition=GPUNodes
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding

container=/logiciels/containerCollections/CUDA10/pytorch.sif
python=/users/samova/lcances/.miniconda3/envs/dl/bin/python
script=../standalone/full_supervised_with_augmentation.py

srun singularity exec ${container} ${python} ${script} -t 1 2 3 4 5 6 7 8 9 -v 10 -T run1 &