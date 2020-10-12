# ___________________________________________________________________________________ #
die() {
    printf '%s\n' "$1" >& 2
    exit 1
}

parse_long() {
    if [ "$1" ]; then
        echo $1
    else
        die "missing argument value"
    fi
}

function show_help {
    echo "usage:  $BASH_SOURCE dataset model [-n | --node] [-N | --nb_task] \
                  [-g | --nb_gpu] [-p | --partition]"
    echo ""
    echo "Mandatory argument"
    echo "    dataset DATASET               Available are {ubs8k, esc10, esc50, speechcommand}"
    echo "    model MODEL                   Available are {cnn03, resnet[18|34|50], wideresnet28_[2|4|8]}"
    echo ""
    echo "Options"
    echo "    -n | --node NODE              On which node the job will be executed"
    echo "    -N | --nb_task NB TASK        On many parallel task"
    echo "    -g | --nb_gpu  NB GPU         On how many gpu this training should be done"
    echo "    -p | --partition PARTITION    On which partition the script will be executed"
    echo ""
    echo "Available partition"
    echo "    GPUNodes"
    echo "    RTX6000Node"
}

# default parameters
NODE=" "
NB_TASK=1
NB_GPU=1
PARTITION="GPUNodes"


# Parse the first two parameters
MODEL=$1; shift;
DATASET=$1; shift;
[[ $MODEL = -?* || $MODEL = "" ]] && die "please provide a model and a dataset"
[[ $DATASET = -?* || $DATASET = "" ]] && die "please provide a dataset"

# Parse the optional parameters
while :; do
    # If no more option (o no option at all)
    if ! [ "$1" ]; then break; fi

    case $1 in
        -n | --node) NODE=$(parse_long $2); shift; shift;;
        -N | --nb_task) NB_TASK=$(parse_long $2); shift; shift;;
        -g | --nb_gpu) NB_GPU=$(parse_long $2); shift; shift;;
        -p | --partition) PARTITION=$(parse_long $2); shift; shift;;

        -?*) echo "WARN: unknown option" $1 >&2
    esac
done

if [ "${NODE}" = " " ]; then
   NODELINE=""
else
    NODELINE="#SBATCH --nodelist=${NODE}"
fi

# ___________________________________________________________________________________ #
LOG_DIR="logs"
SBATCH_JOB_NAME=uniloss_${DATASET}_${MODEL}

cat << EOT > .sbatch_tmp.sh
#!/bin/bash
#SBATCH --job-name=${SBATCH_JOB_NAME}
#SBATCH --output=${LOG_DIR}/${SBATCH_JOB_NAME}.out
#SBATCH --error=${LOG_DIR}/${SBATCH_JOB_NAME}.err
#SBATCH --ntasks=$NB_TASK
#SBATCH --cpus-per-task=5
#SBATCH --partition=$PARTITION
#SBATCH --gres=gpu:$NB_GPU
#SBATCH --gres-flags=enforce-binding
$NODELINE


# sbatch configuration
# container=/logiciels/containerCollections/CUDA10/pytorch.sif
container=/users/samova/lcances/container/pytorch-dev.sif
python=/users/samova/lcances/.miniconda3/envs/pytorch-dev/bin/python
script=../co-training/co-training-uniloss.py

commun_args=""
commun_args="\${commun_args} --supervised_ratio 0.1"
commun_args="\${commun_args} --model ${MODEL}"
commun_args="\${commun_args} --nb_epoch 5000"
commun_args="\${commun_args} --learning_rate 0.0005"
commun_args="\${commun_args} --lambda_cot_max 1"
commun_args="\${commun_args} --lambda_diff_max 1"
commun_args="\${commun_args} --loss_scheduler weighted-linear"
commun_args="\${commun_args} --steps 5000"
commun_args="\${commun_args} --cycle 1"
commun_args="\${commun_args} --beta 1"


echo "commun args"
echo $commun_args

# Run ubs8K models
run_ubs8k() {
    dataset_args="--dataset ubs8k --batch_size 64 --num_classes 10 \
                 -t 1 2 3 4 5 6 7 8 9 -v 10"

    srun -n 1 -N 1 singularity exec \${container} \${python} \${script} \${commun_args} \${dataset_args}
}

# Run esc10 models
run_esc10() {
    dataset_args="--dataset esc10 --batch_size 64 --num_classes 10 \
                 -t 1 2 3 4 -v 5"

    srun -n 1 -N 1 singularity exec \${container} \${python} \${script} \${commun_args} \${dataset_args}
}

# Run esc50 models
run_esc50() {
    dataset_args="--dataset esc50 --batch_size 64 --num_classes 50 \
                 -t 1 2 3 4 -v 5"

    srun -n 1 -N 1 singularity exec \${container} \${python} \${script} \${commun_args} \${dataset_args}
}

# Run speechcommads models
run_speechcommand() {
    dataset_args="--dataset SpeechCommand --batch_size 256 --num_classes 35"

    srun -n 1 -N 1 singularity exec \${container} \${python} \${script} \${commun_args} \${dataset_args}
}

case $DATASET in
    ubs8k) run_ubs8k; exit 0;;
    esc10) run_esc10; exit 0;;
    esc50) run_esc50; exit 0;;
    speechcommand | SpeechCommand) run_speechcommand; exit 0;;
    ?*) die "this dataset is not available"; exit 1;;
esac

EOT

echo "sbatch store in .sbatch_tmp.sh"
sbatch .sbatch_tmp.sh
