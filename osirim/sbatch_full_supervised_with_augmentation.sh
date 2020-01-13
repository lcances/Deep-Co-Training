#!/bin/sh

#SBATCH --job-name=full_supervised_aug
#SBATCH --output=%j_%t_full_supervised_aug.out
#SBATCH --error=%j_%t_full_supervised_aug.err
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=5
#SBATCH --partition=GPUNodes
##SBATCH --nodelist=gpu-nc06
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding
 
container=/logiciels/containerCollections/CUDA10/pytorch.sif
python=/users/samova/lcances/.miniconda3/envs/dl/bin/python
script=../standalone/full_supervised_with_augmentation.py
    
srun -n1 -N1 singularity exec ${container} ${python} ${script} -t 1 2 3 4 5 6 7 8 9 -v 10 -T run1 &
srun -n1 -N1 singularity exec ${container} ${python} ${script} -t 1 2 3 4 5 6 7 8 10 -v 9 -T run2 &
srun -n1 -N1 singularity exec ${container} ${python} ${script} -t 1 2 3 4 5 6 7 9 10 -v 8 -T run3 &
srun -n1 -N1 singularity exec ${container} ${python} ${script} -t 1 2 3 4 5 6 8 9 10 -v 7 -T run4 &
wait
srun -n1 -N1 singularity exec ${container} ${python} ${script} -t 1 2 3 4 5 7 8 9 10 -v 6 -T run5 &
srun -n1 -N1 singularity exec ${container} ${python} ${script} -t 1 2 3 4 6 7 8 9 10 -v 5 -T run6 &
srun -n1 -N1 singularity exec ${container} ${python} ${script} -t 1 2 3 5 6 7 8 9 10 -v 4 -T run7 &
srun -n1 -N1 singularity exec ${container} ${python} ${script} -t 1 2 4 5 6 7 8 9 10 -v 3 -T run8 &
wait
srun -n1 -N1 singularity exec ${container} ${python} ${script} -t 1 3 4 5 6 7 8 9 10 -v 2 -T run9 &
srun -n1 -N1 singularity exec ${container} ${python} ${script} -t 2 3 4 5 6 7 8 9 10 -v 1 -T run10 &
wait
