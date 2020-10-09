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
                  [-g | --nb_gpu] [-p | --partition] [training parameters]"
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
    echo "Training parameters"
    echo "    --batch_size BATCH_SIZE       The batch size"
    echo "    --learning_rate LR            The initial learning rate"
    echo "    --nb_epoch NB_EPOCH           The total number of epoch"
    echo "    --supervised_ratio SR         The ratio of supervised file"
    echo "    --lambda_cot_max LCM          Lambda cot max"
    echo "    --lambda_diff_max LDM         Lambda diff max"
    echo "    --warmup_lenght WL            Warmup lenght"
    echo "    --seed SEED                   The random generation seed"
    echo "    --tensorboard_sufix SUFIX     Sufix for the tensorboard name, more precision"
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

BATCH_SIZE=100
LR=0.003
NB_EPOCH=300
SR=0.1
LCM=10
LDM=0.5
WL=80
SEED=1234
SUFIX=""


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
        
        --batch_size) BATCH_SIZE=$(parse_long $2); shift: shift;; #
        --learning_rate) LR=$(parse_long $2); shift; shift;; #
        --nb_epoch) NB_EPOCH=$(parse_long $2); shift; shift;; #
        --supervised_ratio) SR=$(parse_long $2); shift; shift;; #
        --lambda_cot_max) LCM=$(parse_long $2); shift; shift;; #
        --lambda_diff_max) LDM=$(parse_long $2); shift; shift;; #
        --warmup_length) WL=$(parse_long $2); shift; shift;; #
        --seed) SEED=$(parse_long $2); shift; shift;; #
        --tensorboard_sufix) SUFIX=$(parse_long $2); shift; shift;;

        -?*) echo "WARN: unknown option" $1 >&2; exit 1
    esac
done

if [ "${NODE}" = " " ]; then
   NODELINE=""
else
    NODELINE="#SBATCH --nodelist=${NODE}"
fi

# ___________________________________________________________________________________ #
LOG_DIR="logs"
SBATCH_JOB_NAME=dct_${DATASET}_${MODEL}_${LR}lr_${LCM}lcm_${LDM}ldm_${WL}wl

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
script=../co-training/co-training.py

commun_args=""
commun_args="\${commun_args} --seed ${SEED}"
commun_args="\${commun_args} --model ${MODEL}"
commun_args="\${commun_args} --supervised_ratio ${SR} --learning_rate ${LR}"
commun_args="\${commun_args} --batch_size ${BATCH_SIZE} --nb_epoch ${NB_EPOCH}"
commun_args="\${commun_args} --lambda_cot_max ${LCM} --lambda_diff_max ${LDM} --warmup_length ${WL}"
commun_args="\${commun_args} --tensorboard_path deep-co-training_grid-search"
commun_args="\${commun_args} --tensorboard_sufix ${SUFIX}"

echo "commun args"
echo $commun_args

# Run ubs8K models
run_ubs8k() {
    dataset_args="--dataset ubs8k --num_classes 10 -t 1 2 3 4 5 6 7 8 9 -v 10"

    srun -n 1 -N 1 singularity exec \${container} \${python} \${script} \${commun_args} \${dataset_args}
}

# Run esc10 models
run_esc10() {
    dataset_args="--dataset esc10 --num_classes 10 -t 1 2 3 4 -v 5"

    srun -n 1 -N 1 singularity exec \${container} \${python} \${script} \${commun_args} \${dataset_args}
}

# Run esc50 models
run_esc50() {
    dataset_args="--dataset esc50 --num_classes 50 -t 1 2 3 4 -v 5"

    srun -n 1 -N 1 singularity exec \${container} \${python} \${script} \${commun_args} \${dataset_args}
}

# Run speechcommads models
run_speechcommand() {
    dataset_args="--dataset SpeechCommand --num_classes 35"

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
