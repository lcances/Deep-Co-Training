#!/bin/bash
#SBATCH --job-name=mS_PSC1_075_full_0.40_scallable2
#SBATCH --output=mS_PSC1_075_full_0.40_scallable2.out
#SBATCH --error=mS_PSC1_075_full_0.40_scallable2.err
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=5
#SBATCH --partition=GPUNodes
#SBATCH --nodelist=gpu-nc02
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding

PR=0.40
MODEL=scallable2

# ---- Model ----
parameters="--model ${MODEL}"

# ---- Hyperparaters ----
if [ "$MODEL" = "cnn" ]; then
  hyper_parameters="--base_lr 0.01 --lambda_cot_max 5 --lambda_diff_max 0.25 --warm_up 160 --epsilon 0.1"
fi

if [ "$MODEL" = "scallable2" ]; then
  hyper_parameters="--base_lr 0.01 --lambda_cot_max 2 --lambda_diff_max 0.5 --warm_up 120 --epsilon 0.02"
fi
parameters="${parameters} ${hyper_parameters}"

# ---- parser ratio ----
parameters="${parameters} --parser_ratio ${PR}"

# ---- subsampling ----
#parameters="${parameters} --subsampling 0.1 --subsampling_method balance"

# ---- num_workers ----
# parameters="${parameters} --num_workers 4"

# ---- number of epochs ----
parameters="${parameters} --epochs 400"

# ---- tensorboard ----
parameters="${parameters} --tensorboard_dir moreS_PSC1_075_full"

# ---- log system ----
parameters="${parameters} --log info"

# ---- global augmentation paramters ----
parameters="${parameters} --augment_S" # augmentation is applied on supervised files only

# ---- static augmentation ---- (must be a valid python dictionnary and in the last parameters)
# parameters=${parameters} --static_augments=\"{'PSC1': 0.75}\"

# ---- dynamic augmentation ---- (must always be last)
# aug1='signal_augmentations.Noise(0.75, target_snr=20)'
# parameters=" -a=\"\""


# Sbatch configuration
container=/logiciels/containerCollections/CUDA10/pytorch.sif
python=/users/samova/lcances/.miniconda3/envs/dl/bin/python
script=../standalone/co-training_static_aug.py

folds=(
	"-t 1 2 3 4 5 6 7 8 10 -v 9" 	"-t 1 2 3 4 5 6 7 8 9 -v 10" 	"-t 2 3 4 5 6 7 8 9 10 -v 1" 	"-t 1 3 4 5 6 7 8 9 10 -v 2" 	"-t 1 2 4 5 6 7 8 9 10 -v 3" 	"-t 1 2 3 5 6 7 8 9 10 -v 4" 	"-t 1 2 3 4 6 7 8 9 10 -v 5" 	"-t 1 2 3 4 5 7 8 9 10 -v 6" 	"-t 1 2 3 4 5 6 8 9 10 -v 7" 	"-t 1 2 3 4 5 6 7 9 10 -v 8" )

job_number=0
for i in ${!folds[*]}
do
    job_name="--job_name none_${PR}pr_run${job_number}"
    srun -n1 -N1 singularity exec ${container} ${python} ${script} ${folds[$i]} ${job_name} ${parameters} --static_augments="{'PSC1': 0.75}" &
    
    # automatically wait
    job_number=$(( $job_number + 1 ))
    if [ $(($job_number % 3)) -eq 0 ]
    then
        job_number=0
        wait
    fi
done

wait # needed for the last non NTASK multiple training

