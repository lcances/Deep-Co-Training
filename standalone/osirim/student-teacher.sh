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
    echo "usage:  $BASH_SOURCE dataset model [-r | --ratio] [training options]"
    echo ""
    echo "Mandatory argument"
    echo "    dataset DATASET               Available are {ubs8k, esc{10|05}, speechcommand}"
    echo "    model MODEL                   Available are {cnn03, resnet{18|34|50}, wideresnet28_[2|4|8]}"
    echo ""
    echo "Options"
    echo "    -e  | --epoch EPOCH           The number of epoch"
    echo "    -b  | --batch_size            The size of the batch"
    echo "    -r  | --ratio RATIO           The supervised ratio to use"
    echo "    -lr | --learning_rate         The learning rate" 
    echo "    --lambda_ccost_max LCM        The consistency cost maximum value"
    echo "    --alpha ALPHA                 Alpha value for the exponential moving average"
    echo "    --warmup_length               The length of the warmup"
    echo "    --tensorboard_sufix SUFIX     Sufix for the tensorboard name, more precision"
    echo ""
    echo "Osirim related parameters"
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
EPOCH=200
BS=64
LR=0.003
RATIO="0.1"
LCM=2
ALPHA=0.999
WL=100
SUFIX="_"

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
        -e | --epoch)          EPOCH=$(parse_long $2); shift; shift;;
        -b | --batch_size)     BS=$(parse_long $2); shift; shift;;
        -r | --ratio)          RATIO=$(parse_long $2); shift; shift;;
        -lr | --learning_rate) LR=$(parse_long $2); shift; shift;;
        --lambda_ccost_max)    LCM=$(parse_long $2); shift; shift;;
        --alpha)               ALPHA=$(parse_long $2); shift; shift;;
        --warmup_length)       WL=$(parse_long $2); shift; shift;;
        --tensorboard_sufix) SUFIX=$(parse_long $2); shift; shift;;

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
SBATCH_JOB_NAME=st_${DATASET}_${MODEL}_${RATIO}S

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
script=../student-teacher/student-teacher.py

# ___________________________________________________________________________________ #

commun_args=""
commun_args="\${commun_args} --dataset ${DATASET}"
commun_args="\${commun_args} --model ${MODEL}"

commun_args="\${commun_args} --supervised_ratio ${RATIO}"
commun_args="\${commun_args} --batch_size ${BS}"
commun_args="\${commun_args} --learning_rate ${LR}"
commun_args="\${commun_args} --nb_epoch ${EPOCH}"

commun_args="\${commun_args} --lambda_cost_max ${LCM}"
commun_args="\${commun_args} --warmup_length ${WL}"
commun_args="\${commun_args} --ema_alpha ${ALPHA}"

commun_args="\${commun_args} --tensorboard_sufix ${SUFIX}"


# Run ubs8K models
run_ubs8k() {
    dataset_args="--num_classes 10 -t 1 2 3 4 5 6 7 8 9 -v 10"

    srun -n 1 -N 1 singularity exec \${container} \${python} \${script} \${commun_args} \${dataset_args}
}

# Run esc10 models
run_esc10() {
    dataset_args="--num_classes 10 -t 1 2 3 4 -v 5"

    srun -n 1 -N 1 singularity exec \${container} \${python} \${script} \${commun_args} \${dataset_args}
}

# Run esc50 models
run_esc50() {
    dataset_args="--num_classes 50 -t 1 2 3 4 -v 5"

    srun -n 1 -N 1 singularity exec \${container} \${python} \${script} \${commun_args} \${dataset_args}
}

# Run speechcommads models
run_speechcommand() {
    dataset_args="--num_classes 35"

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
#sbatch .sbatch_tmp.sh
