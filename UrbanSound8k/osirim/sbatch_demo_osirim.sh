#!/bin/bash

#SBATCH --job-name=demo_osirim
#SBATCH --output=demo_osirim.out
#SBATCH --error=demo_osirim.err

#SBATCH --partition=GPUNodes

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5

#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding

container=/logiciels/containerCollections/CUDA10/pytorch.sif
python=/users/samova/lcances/.miniconda3/envs/dl/bin/python
script=full_supervised_demo_osirim.py

srun singularity exec ${container} ${python} ${script}
