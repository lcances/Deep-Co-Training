#!/bin/bash

# Prepare the sbatch --------
#SBATCH --job-name=compound_scale2
#SBATCH --output=compound_scale2.out
#SBATCH --error=compound_scale2.err
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=5
#SBATCH --partition=GPUNodes
#SBATCH --nodelist=gpu-nc05
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding

container=/logiciels/containerCollections/CUDA10/pytorch.sif
python=/users/samova/lcances/.miniconda3/envs/dl/bin/python
script=../../standalone/CompoundScalling/CompoundScalling1.py

# set up initial model architecture
init_conv="--init_conv_inputs 1 44 89 89 --init_conv_outputs 44 89 89 89"
init_linear="--init_linear_inputs 3560 --init_linear_outputs 10"
init_resolution="--init_resolution 64 173"
init_model_param="${init_conv} ${init_linear} ${init_resolution}"

fixed_parameters="--job_name grid_search --train_folds 1 2 3 4 5 6 7 8 9 --val_folds 10 ${init_model_param}"

alpha_list=(1.0 1.2 1.4 1.6 1.8 2.0)
beta_list=(1.0 1.2 1.4 1.6 1.8 2.0)
gamma_list=(1.0)

index=0
ntasks=4
echo $alpha_list

for alpha in ${alpha_list[*]}
do
    for beta in ${beta_list[*]}
    do
        for gamma in ${gamma_list[*]}
        do
            valid=$(echo $alpha*$beta*$beta*$gamma*$gamma | bc -l)

            if [ $(echo "$valid < 2" | bc -l) -eq 1 ]
            then
                factors="-a ${alpha} -b ${beta} -g ${gamma}"

                srun -n1 -N1 singularity exec ${container} ${python} ${script} ${fixed_parameters} ${factors} &

                index=$((index+1))

                if [ $(echo "$index % $ntasks" | bc) -eq 0 ]
                then
                    echo wait
                    wait
                fi
            fi
        done
    done
done
