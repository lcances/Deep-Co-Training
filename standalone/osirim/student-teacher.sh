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
    echo "Miscalleous arguments"
    echo "    -C | --crossval   CROSSVAL (default FALSE)"
    echo "    -R | --resume     RESUME (default FALSE)"
    echo "    -h help"
    echo ""
    echo "Training parameters"
    echo "    --dataset            DATASET (default ubs8k)"
    echo "    --model              MODEL (default wideresnet28_4)"
    echo "    --ratio              SUPERVISED RATIO (default 1.0)"
    echo "    --epoch              EPOCH (default 200)"
    echo "    --learning_rate      LR (default 0.001)"
    echo "    --batch_size         BATCH_SIZE (default 64)"
    echo "    --seed               SEED (default 1234)"
    echo "    --lambda_ccost_max   LCM The consistency cost maximum value"
    echo "    --alpha              ALPHA value for the exponential moving average"
    echo "    --warmup_length      WL The length of the warmup"
    echo "    --noise              NOISE Add noise to the teacher input"
    echo "    --tensorboard_sufix  SUFIX for the tensorboard name, more precision"
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
# osirim parameters
NODE=" "
NB_TASK=1
NB_GPU=1
PARTITION="GPUNodes"

# training parameters
MODEL=cnn03
DATASET="ubs8k"
RATIO=0.1
EPOCH=200
NB_CLS=10
BATCH_SIZE=100
RESUME=0
CROSSVAL=0
LR=0.003
SEED=1234
LCM=2
ALPHA=0.999
WL=100
NOISE=""
SUFIX="_"


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
        -R | --resume)        RESUME=1; shift;;
        -C | --crossval)      CROSSVAL=1; shift;;

        --dataset)            DATASET=$(parse_long $2); shift; shift;;
        --model)              MODEL=$(parse_long $2); shift; shift;;
        --ratio)              RATIO=$(parse_long $2); shift; shift;;
        --epoch)              EPOCH=$(parse_long $2); shift; shift;;
        --learning_rate)      LR=$(parse_long $2); shift; shift;;
        --batch_size)         BATCH_SIZE=$(parse_long $2); shift; shift;;
        --seed)               SEED=$(parse_long $2); shift; shift;;
        --tensorboard_sufix)  SUFIX=$(parse_long $2); shift; shift;;

        --lambda_ccost_max)   LCM=$(parse_long $2); shift; shift;;
        --alpha)              ALPHA=$(parse_long $2); shift; shift;;
        --warmup_length)      WL=$(parse_long $2); shift; shift;;
	--noise)              NOISE="--noise"; shift;;

        -n | --node)      NODE=$(parse_long $2); shift; shift;;
        -N | --nb_task)   NB_TASK=$(parse_long $2); shift; shift;;
        -g | --nb_gpu)    NB_GPU=$(parse_long $2); shift; shift;;
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

# prepare cross validation parameters
# ---- default, no crossvalidation
if [ "$DATASET" = "ubs8k" ]; then
    folds=("-t 1 2 3 4 5 6 7 8 9 -v 10")
elif [ "$DATASET" = "esc10" ] || [ "$DATASET" = "esc50" ]; then
    folds=("-t 1 2 3 4 -v 5")
elif [ "$DATASET" = "SpeechCommand" ]; then
    folds=("-t 1 -v 2") # fake array to just ensure at least one run. Not used by the dataset anyway
fi

# if crossvalidation is activated
if [ $CROSSVAL -eq 1 ]; then
    if [ "$DATASET" = "ubs8k" ]; then
        mvar=\$(python -c "import DCT.util.utils as u; u.create_bash_crossvalidation(10)")
        IFS=";" read -a folds <<< \$mvar

    elif [ "$DATASET" = "esc10" ] || [ "$DATASET" = "esc50" ]; then
        mvar=\$(python -c "import DCT.util.utils as u; u.create_bash_crossvalidation(5)")
        IFS=";" read -a folds <<< \$mvar
    fi
fi


# -------- dataset & model ------
common_args="\${common_args} --dataset ${DATASET}"
common_args="\${common_args} --model ${MODEL}"

# -------- training common_args --------
common_args="\${common_args} --supervised_ratio ${RATIO}"
common_args="\${common_args} --nb_epoch ${EPOCH}"
common_args="\${common_args} --learning_rate ${LR}"
common_args="\${common_args} --batch_size ${BATCH_SIZE}"
common_args="\${common_args} --seed ${SEED}"

common_args="\${common_args} --lambda_cost_max ${LCM}"
common_args="\${common_args} --warmup_length ${WL}"
common_args="\${common_args} --ema_alpha ${ALPHA}"
common_args="\${common_args} ${NOISE}"

# -------- resume training --------
if [ $RESUME -eq 1 ]; then
    echo "$RESUME"
    common_args="\${common_args} --resume"
fi

# -------- dataset specific parameters --------
case $DATASET in
    ubs8k | esc10) dataset_args="--num_classes 10";;
    esc50) dataset_args="--num_classes 50";;
    SpeechCommand) dataset_args="--num_classes 35";;
    ?*) die "dataset ${DATASET} is not available"; exit 1;;
esac


run_number=0
for i in \${!folds[*]}
do
    run_number=\$(( \$run_number + 1 ))

    if [ $CROSSVAL -eq 1 ]; then
        tensorboard_sufix="--tensorboard_sufix run\${run_number}"
    else
        tensorboard_sufix=""
    fi

    extra_params="\${tensorboard_sufix} \${folds[\$i]}"
    
    echo srun -n 1 -N 1 singularity exec \${container} \${python} \${script} \${common_args} \${dataset_args} \${extra_params}
    srun -n 1 -N 1 singularity exec \${container} \${python} \${script} \${common_args} \${dataset_args} \${extra_params}
done


EOT

echo "sbatch store in .sbatch_tmp.sh"
sbatch .sbatch_tmp.sh
