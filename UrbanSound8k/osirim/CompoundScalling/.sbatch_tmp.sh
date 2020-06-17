#!/bin/bash

# Prepare the sbatch --------
#SBATCH --job-name=CS_step2_False
#SBATCH --output=CS_step2_False.out
#SBATCH --error=CS_step2_False.err
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=5
#SBATCH --partition=GPUNodes
#SBATCH --nodelist=gpu-nc07
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding

container=/logiciels/containerCollections/CUDA10/pytorch.sif
python=/users/samova/lcances/.miniconda3/envs/dl/bin/python
script=../../standalone/CompoundScalling/copoundScalling1.py

# List of scale that fit the following constrain: a * b² * y² ~= 2
scales=(
    "-a 1.1579 -b 1.0000 -g 1.3158"     "-a 1.1579 -b 1.3158 -g 1.0000"     "-a 1.3684 -b 1.0000 -g 1.2105"     "-a 1.3684 -b 1.2105 -g 1.0000"     "-a 1.4737 -b 1.0526 -g 1.1053"     "-a 1.4737 -b 1.1053 -g 1.0526"     "-a 1.6316 -b 1.0000 -g 1.1053"     "-a 1.6316 -b 1.0526 -g 1.0526"     "-a 1.6316 -b 1.1053 -g 1.0000"     "-a 2.0000 -b 1.0000 -g 1.0000" )


job_number=0
for s in ${!scales[*]}
do
    srun -n1 -N1 singularity exec ${container} ${python} ${script} ${scales[$s]} --round_up False --title step2_False &
      
    # automatically wait
    job_number=$(( $job_number + 1 ))
    if [ $(($job_number % 4)) -eq 0 ]
    then
        job_number=0
        wait
    fi
done

