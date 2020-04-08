#!/bin/bash

#SBATCH --job-name=crossval_dct_moreS
#SBATCH --output=crossval_dct_moreS_%j.out
#SBATCH --error=crossval_dct_moreS_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --partition=GPUNodes
##SBATCH --nodelist=gpu-nc06
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding

container=/logiciels/containerCollections/CUDA10/pytorch.sif
python=/users/samova/lcances/.miniconda3/envs/dl/bin/python
script=../standalone/co-training.py

if [ "$#" -ne 2 ]; then
  echo "wrong number of parameter"
  echo "received $# arguments"
  echo "usage: [parser_ratio] [model]"
  exit 1
fi

parser_ratio="--parser_ratio $1"
model="--model $2"

if [ "$2" = "cnn" ]; then
  hyper_parameters="--base_lr 0.01 --lambda_cot_max 5 --lambda_diff_max 0.25 --warm_up 160 --epsilon 0.1"
fi

if [ "$2" = "scallable2" ]; then
  hyper_parameters="--base_lr 0.01 --lambda_cot_max 2 --lambda_diff_max 0.5 --warm_up 120 --epsilon 0.02"
fi

# global parameters
subsampling="--subsampling 0.1 --subsampling_method balance"
parameters="${model} ${parser_ratio} ${hyper_parameters} ${subsampling} -T moreS_ss1.0"

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
for i in ${!folds[*]}
do
  job_name="--job_name none_$1pr_run${job_number}"
  srun -n1 -N1 singularity exec ${container} ${python} ${script} ${parameters} ${folds[$i]} ${job_name}"
  job_number=$(( $job_number + 1 ))
done
