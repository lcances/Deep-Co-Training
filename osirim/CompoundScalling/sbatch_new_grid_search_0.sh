#!/bin/bash

# Prepare the sbatch --------
#SBATCH --job-name=compound_scale0
#SBATCH --output=compound_scale0.out
#SBATCH --error=compound_scale0.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --partition=RTX6000Node
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding

container=/logiciels/containerCollections/CUDA10/pytorch.sif
python=/users/samova/lcances/.miniconda3/envs/dl/bin/python
script=../../standalone/CompoundScalling/compoundScalling0.py

# List of scale that fit the following constrain: a * b² * y² ~= 2
scales=(
    "-a 1.1579 -b 1.0000 -g 1.3158" \
    "-a 1.1579 -b 1.3158 -g 1.0000" \
    "-a 1.3684 -b 1.0000 -g 1.2105" \
    "-a 1.3684 -b 1.2105 -g 1.0000" \
    "-a 1.4737 -b 1.0526 -g 1.1053" \
    "-a 1.4737 -b 1.1053 -g 1.0526" \
    "-a 1.6316 -b 1.0000 -g 1.1053" \
    "-a 1.6316 -b 1.0526 -g 1.0526" \
    "-a 1.6316 -b 1.1053 -g 1.0000" \
    "-a 2.0000 -b 1.0000 -g 1.0000" \
)

folds=(
	"-t 2 3 4 5 6 7 8 9 10 -v 1" \
	"-t 1 3 4 5 6 7 8 9 10 -v 2" \
	"-t 1 2 4 5 6 7 8 9 10 -v 3" \
	"-t 1 2 3 5 6 7 8 9 10 -v 4" \
	"-t 1 2 3 4 6 7 8 9 10 -v 5" \
	"-t 1 2 3 4 5 7 8 9 10 -v 6" \
	"-t 1 2 3 4 5 6 8 9 10 -v 7" \
	"-t 1 2 3 4 5 6 7 9 10 -v 8" \
	"-t 1 2 3 4 5 6 7 8 10 -v 9" \
	"-t 1 2 3 4 5 6 7 8 9 -v 10" \
)



job_number=1
for s in ${!scales[*]}
do
    for i in ${!folds[*]}
    do
      job_name="--run_number ${job_number}"
      srun -n1 -N1 singularity exec ${container} ${python} ${script} ${job_name} ${folds[$i]} ${scales[$s]}
      job_number=$(( $job_number + 1 ))
    done
done
