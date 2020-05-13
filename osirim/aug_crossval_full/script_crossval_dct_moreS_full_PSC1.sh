#!/bin/bash

if [ "$#" -ne 3 ]; then
  echo "wrong number of parameter"
  echo "received $# arguments"
  echo "usage: gpu-nc0x parser_ratio model"
  exit 1
fi

SBATCH_JOB_NAME=mS_PSC1_$2_$3

cat << EOT > .sbatch_tmp.sh
#!/bin/bash
#SBATCH --job-name=$SBATCH_JOB_NAME
#SBATCH --output=${SBATCH_JOB_NAME}_%j.out
#SBATCH --error=${SBATCH_JOB_NAME}_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --partition=GPUNodes
#SBATCH --nodelist=$1
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding

PR=$2
MODEL=$3

parser_ratio="--parser_ratio \${PR}"
model="--model \${MODEL}"

if [ "\$MODEL" = "cnn" ]; then
  hyper_parameters="--base_lr 0.01 --lambda_cot_max 5 --lambda_diff_max 0.25 --warm_up 160 --epsilon 0.1"
fi

if [ "\$MODEL" = "scallable2" ]; then
  hyper_parameters="--base_lr 0.01 --lambda_cot_max 2 --lambda_diff_max 0.5 --warm_up 120 --epsilon 0.02"
fi

# augmentation
aug1="signal_augmentations.PitchShiftChoice(0.5, choice=(-3, -2, 2, 3))"

# global parameters
augmentation="--augment_S"
parameters="\${model} \${parser_ratio} \${hyper_parameters} \${augmentation} --num_workers 4 --epochs 400 -T moreS_full_PSC1 --log info"

# Sbatch configuration
container=/logiciels/containerCollections/CUDA10/pytorch.sif
python=/users/samova/lcances/.miniconda3/envs/dl/bin/python
script=../standalone/co-training.py
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
for i in \${!folds[*]}
do
  job_name="--job_name none_\${PR}pr_run\${job_number}"
  srun -n1 -N1 singularity exec \${container} \${python} \${script} \${parameters} \${folds[\$i]} \${job_name} -a="\${aug1}"
  job_number=\$(( \$job_number + 1 ))
done

EOT

echo "sbatch store in .sbatch_tmp.sh"
sbatch .sbatch_tmp.sh
