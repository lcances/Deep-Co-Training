#!/bin/bash

# ___________________________________________________________________________________ #
parse_long() {
    if [ "$1" ]; then
        echo $1
    else
        echo "Missing argument value" >&2
        exit 1
    fi
}

function show_help {
    echo "usage:  $BASH_SOURCE [--dataset] [--model] [--ratio] [--epoch] [--lambda_sup_max] [--lambda_cot_max] [--lambda_diff_max] \
    [--crossval] [-R] [-h]"
    echo "    --dataset DATASET (default ubs8k)"
    echo "    --model MODEL (default cnn03)"
    echo "    --ratio SUPERVISED RATIO (default 0.1)"
    echo "    --epoch EPOCH (default 200)"
    echo ""
    echo "Co-training hyperparameters"
    echo "    --lambda_sup_max LAMBDA SUP MAX (default 1)"
    echo "    --lambda_cot_max LAMBDA COT MAX (default 10)"
    echo "    --lambda_diff_max LAMBDA_DIFF_MAX (default 0.5)"
    echo ""
    echo "Miscalleous arguments"
    echo "    --crossval (default FALSE)"
    echo "    -R RESUME (default FALSE)"
    echo "    -h help"
    
    echo "Available datasets"
    echo "    ubs8k"
    echo "    cifar10"
    
    echo "Available models"
    echo "    PModel"
    echo "    wideresnet28_2"
    echo "    cnn0"
    echo "    cnn03"
    echo "    scallable1"
}

# default parameters
DATASET="ubs8k"
MODEL=cnn03
RATIO=0.1
EPOCH=200
LSM=1
LCM=10
LDM=0.5
RESUME=0
CROSSVAL=0

while :; do
    case $1 in
        -h | -\? | --help) show_help; exit 1;;
        -R) RESUME=1; SHIFT;;
        --crossval) CROSSVAL=1; shift;;

        --dataset) DATASET=$(parse_long $2); shift; shift;;
        --model) MODEL=$(parse_long $2); shift; shift;;
        --ratio) RATIO=$(parse_long $2); shift; shift;;
        --epoch) EPOCH=$(parse_long $2); shift; shift;;
        --lambda_sup_max) LSM=$(parse_long $2); shift; shift;;
        --lambda_cot_max) LCM=$(parse_long $2); shift; shift;;
        --lambda_diff_max) LDM=$(parse_long $2); shift; shift;;

        -?*) echo "Invalide option $1" >&2; show_help; exit 1;;
    esac
done

# ___________________________________________________________________________________ #
# ___________________________________________________________________________________ #

if [ $CROSSVAL -eq 1 ]; then
    echo "Cross validation activated"
    folds=(
        "-t 1 2 3 4 5 6 7 8 9 -v 10" \
        "-t 2 3 4 5 6 7 8 9 10 -v 1" \
        "-t 1 3 4 5 6 7 8 9 10 -v 2" \
        "-t 1 2 4 5 6 7 8 9 10 -v 3" \
        "-t 1 2 3 5 6 7 8 9 10 -v 4" \
        "-t 1 2 3 4 6 7 8 9 10 -v 5" \
        "-t 1 2 3 4 5 7 8 9 10 -v 6" \
        "-t 1 2 3 4 5 6 8 9 10 -v 7" \
        "-t 1 2 3 4 5 6 7 9 10 -v 8" \
        "-t 1 2 3 4 5 6 7 8 10 -v 9" \
    )
else
    folds=("-t 1 2 3 4 5 6 7 8 9 -v 10")
fi

tensorboard_path_root="--tensorboard_path ../../tensorboard/${DATASET}/deep-co-training"
checkpoint_path_root="--checkpoint_path ../../model_save/${DATASET}/deep-co-training"

# ___________________________________________________________________________________ #
parameters=""

# -------- tensorboard and checkpoint path --------
tensorboard_path="${tensorboard_path_root}/${MODEL}/${RATIO}S"
checkpoint_path="${checkpoint_path_root}/${MODEL}/${RATIO}S"
parameters="${parameters} ${tensorboard_path} ${checkpoint_path}"

# -------- model --------
parameters="${parameters} --model ${MODEL}"

# -------- training parameters --------
parameters="${parameters} --supervised_ratio ${RATIO}"
parameters="${parameters} --nb_epoch ${NB_EPOCH}"

parameters="${parameters} --lambda_sup_max ${LSM}"
parameters="${parameters} --lambda_cot_max ${LCM}"
parameters="${parameters} --lambda_diff_max ${LDM}"

# -------- resume training --------
if [ $RESUME -eq 1 ]; then
    echo "$RESUME"
    parameters="${parameters} --resume"
fi

run_number=0
for i in ${!folds[*]}
do
    run_number=$(( $run_number + 1 ))
    tensorboard_sufix="--tensorboard_sufix run${run_number}"
    
    echo python ${SCRIPT} ${folds[$i]} ${tensorboard_sufix} ${parameters}
    python ${SCRIPT} ${folds[$i]} ${tensorboard_sufix} ${parameters}
done
