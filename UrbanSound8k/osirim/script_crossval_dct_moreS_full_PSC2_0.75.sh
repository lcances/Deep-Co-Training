#!/bin/bash

function show_help {
    echo "usage:  $BASH_SOURCE -n <node_name> -N <ntask> -p <parser_ratio> -m <model_name>"
    echo "    -n NODE_NAME "
    echo "    -N NTASK "
    echo "    -p PARSER_RATIO "
    echo "    -m MODEL "
    echo "    -s SCRIPT (co-training_static_aug.py) "
    echo "    -i CUSTOM_PREFIX_ID "
}

# default parameters
SCRIPT=../standalone/co-training_static_aug.py
ID=""

while getopts ":n:N:p:m:s:i:" arg; do
  case $arg in
    n) NODELIST=$OPTARG;;
    N) NTASK=$OPTARG;;
    p) PARSER_RATIO=$OPTARG;;
    m) MODEL=$OPTARG;;
    s) SCRIPT=$OPTARG;;
    i) ID=${OPTARG}_;;
    *) 
        echo "invalide option" 1>&2
        show_help
        exit 1
        ;;
  esac
done

if [ $OPTIND -ne 9 ] && [ $OPTIND -ne 11 ] && [ $OPTIND -ne 13 ]
then
    echo "missing arguments. $OPTIND"
    show_help
    exit 1
fi    

echo $NODELIST
echo $NTASK
echo $PARSER_RATIO
echo $MODEL
echo $SCRIPT


AUG_IDENTIFIER=PSC2_075_full
SBATCH_JOB_NAME=mS_${ID}${AUG_IDENTIFIER}_${PARSER_RATIO}_${MODEL}

cat << EOT > .sbatch_tmp.sh
#!/bin/bash
#SBATCH --job-name=$SBATCH_JOB_NAME
#SBATCH --output=${SBATCH_JOB_NAME}.out
#SBATCH --error=${SBATCH_JOB_NAME}.err
#SBATCH --ntasks=$NTASK
#SBATCH --cpus-per-task=5
#SBATCH --partition=GPUNodes
#SBATCH --nodelist=$1
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding

PR=$2
MODEL=$3

# ---- Model ----
parameters="--model \${MODEL}"

# ---- Hyperparaters ----
if [ "\$MODEL" = "cnn" ]; then
  hyper_parameters="--base_lr 0.01 --lambda_cot_max 5 --lambda_diff_max 0.25 --warm_up 160 --epsilon 0.1"
fi

if [ "\$MODEL" = "scallable2" ]; then
    hyper_parameters="--base_lr 0.01 --lambda_cot_max 2 --lambda_diff_max 0.5 --warm_up 120 --epsilon 0.02"
fi
parameters="\${parameters} \${hyper_parameters}"

# ---- parser ratio ----
parameters="\${parameters} --parser_ratio \${PR}"

# ---- subsampling ----
# parameters="\${parameters} --subsampling 0.1 --subsampling_method balance"

# ---- num_workers ----
# parameters="\${parameters} --num_workers 4"

# ---- number of epochs ----
parameters="\${parameters} --epochs 400"

# ---- tensorboard ----
parameters="\${parameters} --tensorboard_dir moreS_${AUG_IDENTIFIER}"

# ---- log system ----
parameters="\${parameters} --log info"

# ---- global augmentation paramters ----
parameters="\${parameters} --augment_S" # augmentation is applied on supervised files only

# ---- static augmentation ---- (must be a valid python dictionnary and in the last parameters)
# parameters=\${parameters} --static_augments=\"{'PSC1': 0.75}\"

# ---- dynamic augmentation ---- (must always be last)
# aug1='signal_augmentations.Noise(0.75, target_snr=20)'
# parameters="${parameters} -a=\"${aug1}\""


# Sbatch configuration
container=/logiciels/containerCollections/CUDA10/pytorch.sif
python=/users/samova/lcances/.miniconda3/envs/dl/bin/python
script=${SCRIPT}

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

job_number=0
for i in \${!folds[*]}
do
    job_name="--job_name none_\${PR}pr_run\${job_number}"
    srun -n1 -N1 singularity exec \${container} \${python} \${script} \${folds[\$i]} \${job_name} \${parameters} --static_augments="{'PSC2': 0.75}" &
    
    # automatically wait
    job_number=\$(( \$job_number + 1 ))
    if [ \$((\$job_number % $NTASK)) -eq 0 ]
    then
        job_number=0
        wait
    fi

wait # needed for the last non NTASK multiple training

done

EOT

echo "sbatch store in .sbatch_tmp.sh"
sbatch .sbatch_tmp.sh
