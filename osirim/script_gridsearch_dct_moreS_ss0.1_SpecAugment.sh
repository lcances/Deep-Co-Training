#!/bin/bash

if [ "$#" -ne 3 ]; then
  echo "wrong number of parameter"
  echo "received $# arguments"
  echo "usage: gpu-nc0x parser_ratio model"
  exit 1
fi

ID="gs_SpAu"
SBATCH_JOB_NAME=mS_${ID}_$2_$3

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
augs=(
	"spec_augmentations.FractalDropout(1.0, freq_min_chunk_size=1, freq_max_chunk_size=2, freq_min_chunk=1, freq_max_chunk=2, time_min_chunk_size=1, time_max_chunk_size=6, time_min_chunk=1, time_max_chunk=2)" \
	"spec_augmentations.FractalDropout(1.0, freq_min_chunk_size=1, freq_max_chunk_size=2, freq_min_chunk=1, freq_max_chunk=2, time_min_chunk_size=1, time_max_chunk_size=8, time_min_chunk=1, time_max_chunk=2)" \
	"spec_augmentations.FractalDropout(1.0, freq_min_chunk_size=1, freq_max_chunk_size=2, freq_min_chunk=1, freq_max_chunk=2, time_min_chunk_size=1, time_max_chunk_size=15, time_min_chunk=1, time_max_chunk=2)" \
	"spec_augmentations.FractalDropout(1.0, freq_min_chunk_size=1, freq_max_chunk_size=2, freq_min_chunk=1, freq_max_chunk=2, time_min_chunk_size=1, time_max_chunk_size=20, time_min_chunk=1, time_max_chunk=2)" \
	"spec_augmentations.FractalDropout(1.0, freq_min_chunk_size=1, freq_max_chunk_size=3, freq_min_chunk=1, freq_max_chunk=2, time_min_chunk_size=1, time_max_chunk_size=6, time_min_chunk=1, time_max_chunk=2)" \
	"spec_augmentations.FractalDropout(1.0, freq_min_chunk_size=1, freq_max_chunk_size=3, freq_min_chunk=1, freq_max_chunk=2, time_min_chunk_size=1, time_max_chunk_size=8, time_min_chunk=1, time_max_chunk=2)" \
	"spec_augmentations.FractalDropout(1.0, freq_min_chunk_size=1, freq_max_chunk_size=3, freq_min_chunk=1, freq_max_chunk=2, time_min_chunk_size=1, time_max_chunk_size=15, time_min_chunk=1, time_max_chunk=2)" \
	"spec_augmentations.FractalDropout(1.0, freq_min_chunk_size=1, freq_max_chunk_size=3, freq_min_chunk=1, freq_max_chunk=2, time_min_chunk_size=1, time_max_chunk_size=20, time_min_chunk=1, time_max_chunk=2)" \
	"spec_augmentations.FractalDropout(1.0, freq_min_chunk_size=1, freq_max_chunk_size=4, freq_min_chunk=1, freq_max_chunk=2, time_min_chunk_size=1, time_max_chunk_size=6, time_min_chunk=1, time_max_chunk=2)" \
	"spec_augmentations.FractalDropout(1.0, freq_min_chunk_size=1, freq_max_chunk_size=4, freq_min_chunk=1, freq_max_chunk=2, time_min_chunk_size=1, time_max_chunk_size=8, time_min_chunk=1, time_max_chunk=2)" \
	"spec_augmentations.FractalDropout(1.0, freq_min_chunk_size=1, freq_max_chunk_size=4, freq_min_chunk=1, freq_max_chunk=2, time_min_chunk_size=1, time_max_chunk_size=15, time_min_chunk=1, time_max_chunk=2)" \
	"spec_augmentations.FractalDropout(1.0, freq_min_chunk_size=1, freq_max_chunk_size=4, freq_min_chunk=1, freq_max_chunk=2, time_min_chunk_size=1, time_max_chunk_size=20, time_min_chunk=1, time_max_chunk=2)" \
	"spec_augmentations.FractalDropout(1.0, freq_min_chunk_size=1, freq_max_chunk_size=6, freq_min_chunk=1, freq_max_chunk=2, time_min_chunk_size=1, time_max_chunk_size=6, time_min_chunk=1, time_max_chunk=2)" \
	"spec_augmentations.FractalDropout(1.0, freq_min_chunk_size=1, freq_max_chunk_size=6, freq_min_chunk=1, freq_max_chunk=2, time_min_chunk_size=1, time_max_chunk_size=8, time_min_chunk=1, time_max_chunk=2)" \
	"spec_augmentations.FractalDropout(1.0, freq_min_chunk_size=1, freq_max_chunk_size=6, freq_min_chunk=1, freq_max_chunk=2, time_min_chunk_size=1, time_max_chunk_size=15, time_min_chunk=1, time_max_chunk=2)" \
	"spec_augmentations.FractalDropout(1.0, freq_min_chunk_size=1, freq_max_chunk_size=6, freq_min_chunk=1, freq_max_chunk=2, time_min_chunk_size=1, time_max_chunk_size=20, time_min_chunk=1, time_max_chunk=2)" \
)
# global parameters
subsampling="--subsampling 0.1 --subsampling_method balance"
fold="-t 2 3 4 5 6 7 8 9 10 -v 1"
augmentation="--augment_S"
tensorboard="-T gridSearch_ss0.1_${ID}"
epochs="--epochs 400"
workers="--num_workers 4"
parameters="\${model} \${fold} \${parser_ratio} \${hyper_parameters} \${subsampling} \${augmentation} \${workers} \${epochs} \${tensorboard} --log info"

# Sbatch configuration
container=/logiciels/containerCollections/CUDA10/pytorch.sif
python=/users/samova/lcances/.miniconda3/envs/dl/bin/python
script=../standalone/co-training.py

job_number=1
for i in \${!augs[*]}
do
  job_name="--job_name none_\${PR}pr_run\${job_number}"
  srun -n1 -N1 singularity exec \${container} \${python} \${script} \${parameters} \${job_name} -a="\${augs[\$i]}"
  job_number=\$(( \$job_number + 1 ))
done

EOT

echo "sbatch store in .sbatch_tmp.sh"
#bash .sbatch_tmp.sh
sbatch .sbatch_tmp.sh
