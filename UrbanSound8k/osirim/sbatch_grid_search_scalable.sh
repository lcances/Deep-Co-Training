#!/bin/sh

# Prepare the sbatch --------
#SBATCH --job-name=50_grid_cdt
#SBATCH --output=50_grid_cdt.out
#SBATCH --error=50_grid_cdt.err
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=5
#SBATCH --partition=GPUNodes
##SBATCH --nodelist=gpu-nc05
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding

container=/logiciels/containerCollections/CUDA10/pytorch.sif
python=/users/samova/lcances/.miniconda3/envs/dl/bin/python
script=../standalone/co-training_scallable2.py
fixed_parameters="--job_name scallable_50p --train_folds 1 2 3 4 5 6 7 8 9 --val_folds 10 --ratio 0.5 --epochs 300"
tensorboard_root="--tensorboard_dir tensorboard/0.5r_scallable2_grid_search"

#SIMPLE GRID SEARCH ==================================================================================================
# Change lambda diff max
# srun -n1 singularity exec ${container} ${python} ${script} ${fixed_parameters} ${tensorboard_root}/lambda_diff_max  --lambda_diff_max 0.5 &
# srun -n1 singularity exec ${container} ${python} ${script} ${fixed_parameters} ${tensorboard_root}/lambda_diff_max  --lambda_diff_max 1.0 &
# wait
# srun -n1 singularity exec ${container} ${python} ${script} ${fixed_parameters} ${tensorboard_root}/lambda_diff_max  --lambda_diff_max 2.0 &
# srun -n1 singularity exec ${container} ${python} ${script} ${fixed_parameters} ${tensorboard_root}/lambda_diff_max  --lambda_diff_max 0.1 &
# wait
# srun -n1 singularity exec ${container} ${python} ${script} ${fixed_parameters} ${tensorboard_root}/lambda_diff_max  --lambda_diff_max 0.2 &
# srun -n1 singularity exec ${container} ${python} ${script} ${fixed_parameters} ${tensorboard_root}/lambda_diff_max  --lambda_diff_max 0.25 &

# change lambda cot max
# wait
# srun -n1 singularity exec ${container} ${python} ${script} ${fixed_parameters} ${tensorboard_root}/lambda_cot_max  --lambda_cot_max 2 &
# srun -n1 singularity exec ${container} ${python} ${script} ${fixed_parameters} ${tensorboard_root}/lambda_cot_max  --lambda_cot_max 5 &
# srun -n1 singularity exec ${container} ${python} ${script} ${fixed_parameters} ${tensorboard_root}/lambda_cot_max  --lambda_cot_max 10 &
# srun -n1 singularity exec ${container} ${python} ${script} ${fixed_parameters} ${tensorboard_root}/lambda_cot_max  --lambda_cot_max 13 &
# wait
# srun -n1 singularity exec ${container} ${python} ${script} ${fixed_parameters} ${tensorboard_root}/lambda_cot_max  --lambda_cot_max 15 &
# 
# # change warmup
# srun -n1 singularity exec ${container} ${python} ${script} ${fixed_parameters} ${tensorboard_root}/warmup  --warm_up 40 &
# srun -n1 singularity exec ${container} ${python} ${script} ${fixed_parameters} ${tensorboard_root}/warmup  --warm_up 60 &
# srun -n1 singularity exec ${container} ${python} ${script} ${fixed_parameters} ${tensorboard_root}/warmup  --warm_up 80 &
# wait
# srun -n1 singularity exec ${container} ${python} ${script} ${fixed_parameters} ${tensorboard_root}/warmup  --warm_up 100 &
# srun -n1 singularity exec ${container} ${python} ${script} ${fixed_parameters} ${tensorboard_root}/warmup  --warm_up 120 &
# srun -n1 singularity exec ${container} ${python} ${script} ${fixed_parameters} ${tensorboard_root}/warmup  --warm_up 140 &
# srun -n1 singularity exec ${container} ${python} ${script} ${fixed_parameters} ${tensorboard_root}/warmup  --warm_up 160 &
# 
# # Change epsilon
# wait
# srun -n1 singularity exec ${container} ${python} ${script} ${fixed_parameters} ${tensorboard_root}/epsilon  --epsilon 0.02 &
# srun -n1 singularity exec ${container} ${python} ${script} ${fixed_parameters} ${tensorboard_root}/epsilon  --epsilon 0.05 &
# srun -n1 singularity exec ${container} ${python} ${script} ${fixed_parameters} ${tensorboard_root}/epsilon  --epsilon 0.1 & 
# srun -n1 singularity exec ${container} ${python} ${script} ${fixed_parameters} ${tensorboard_root}/epsilon  --epsilon 0.5 &
# wait
# srun -n1 singularity exec ${container} ${python} ${script} ${fixed_parameters} ${tensorboard_root}/epsilon  --epsilon 1.0 &
# srun -n1 singularity exec ${container} ${python} ${script} ${fixed_parameters} ${tensorboard_root}/epsilon  --epsilon 2.0 &
# 
# # Change base lr
# srun -n1 singularity exec ${container} ${python} ${script} ${fixed_parameters} ${tensorboard_root}/learning_rate  --base_lr 0.001 &
# srun -n1 singularity exec ${container} ${python} ${script} ${fixed_parameters} ${tensorboard_root}/learning_rate  --base_lr 0.01 &
# wait
# srun -n1 singularity exec ${container} ${python} ${script} ${fixed_parameters} ${tensorboard_root}/learning_rate  --base_lr 0.02 &
# srun -n1 singularity exec ${container} ${python} ${script} ${fixed_parameters} ${tensorboard_root}/learning_rate  --base_lr 0.03 &
# srun -n1 singularity exec ${container} ${python} ${script} ${fixed_parameters} ${tensorboard_root}/learning_rate  --base_lr 0.04 &
# srun -n1 singularity exec ${container} ${python} ${script} ${fixed_parameters} ${tensorboard_root}/learning_rate  --base_lr 0.05 &
# wait
# srun -n1 singularity exec ${container} ${python} ${script} ${fixed_parameters} ${tensorboard_root}/learning_rate  --base_lr 0.06 &
# srun -n1 singularity exec ${container} ${python} ${script} ${fixed_parameters} ${tensorboard_root}/learning_rate  --base_lr 0.07 &
# srun -n1 singularity exec ${container} ${python} ${script} ${fixed_parameters} ${tensorboard_root}/learning_rate  --base_lr 0.08 &
# srun -n1 singularity exec ${container} ${python} ${script} ${fixed_parameters} ${tensorboard_root}/learning_rate  --base_lr 0.09 &
# wait
# srun -n1 singularity exec ${container} ${python} ${script} ${fixed_parameters} ${tensorboard_root}/learning_rate  --base_lr 0.10 &
# wait


# COMBINE GRID SEARCH ==================================================================================================
# After analysis, I decided to fix few value
lambda_cot_max=2
warmup=120
epsilon=0.02
fixed_values="--epsilon ${epsilon} --lambda_cot_max ${lambda_cot_max} --warm_up ${warmup}"

# for osirim management
index=0
ntasks=4
 
# # ==== If one variables ====
# variable=(0.001, 0.01, 0.02, 0.03, 0.05)
# for v in ${variable[*]}
# do
#     srun -n1 -N1 singularity exec ${container} ${python} ${script} ${fixed_parameters} ${fixed_values} --tensorboard_dir tensorboard/combination --base_lr ${v} &

#     index=$((index+1))
#     if [ $(echo "$index % $ntasks" | bc) -eq 0 ]
#     then
#         echo wait
#         wait
#     fi
# done


# ==== If two variables ====
lambda_diff_max=(0.1 0.5)
learning_rate=(0.01 0.03 0.1)

for lambda in ${lambda_diff_max[*]}
do
    for lr in ${learning_rate[*]}
    do
        srun -n1 -N1 singularity exec ${container} ${python} ${script} ${fixed_parameters} ${tensorboard_root}/combination ${fixed_values} --base_lr ${lr} --lambda_diff_max ${lambda} &

        index=$((index+1))
        if [ $(echo "$index % $ntasks" | bc) -eq 0 ]
        then
            echo wait
            wait
        fi
done
done


# After this step, the following parameters should be the best
# base_lr = 0.01
# epsilon = 0.1
# lambda_cot_max = 5
# lambda_diff_max = 0.25
# warmup = 160
