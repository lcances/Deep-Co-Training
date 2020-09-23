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
    echo "usage:  $BASH_SOURCE [--dataset] [--model] [--ratio] [--epoch] [--learning_rate] \
        [--batch_size] [--num_classes] [-C | --crossval] [-R | --resume] [-h]"
    echo "    --dataset         DATASET (default ubs8k)"
    echo "    --model           MODEL (default wideresnet28_4)"
    echo "    --ratio           SUPERVISED RATIO (default 1.0)"
    echo "    --epoch           EPOCH (default 200)"
    echo "    --learning_rate   LR (default 0.001)"
    echo "    --batch_size      BATCH_SIZE (default 64)"
    echo "    --num_classes     NB_CLS (default 10)"
    echo ""
    echo ""
    echo "Miscalleous arguments"
    echo "    -C | --crossval   CROSSVAL (default FALSE)"
    echo "    -R | --resume     RESUME (default FALSE)"
    echo "    -h help"
    
    echo "Available datasets"
    echo "    ubs8k"
    echo "    cifar10"
    echo "    esc10"
    echo "    esc50"
    
    echo "Available models"
    echo "    PModel, cnn03, resnet18, resnet34, resnet50, wideresnet28_2, esc_wideresnet28_2, esc_wideresnet28_4, esc_wideresnet28_8"
}

# default parameters
DATASET="ubs8k"
MODEL=cnn03
RATIO=1.0
EPOCH=200
NB_CLS=10
BATCH_SIZE=64
RESUME=0
CROSSVAL=0
LR=0.001

while :; do
    case $1 in
        -h | -\? | --help) show_help; exit 1;;
        -R | --resume)      RESUME=1; shift;;
        -C | --crossval)    CROSSVAL=1; shift;;

        --dataset)          DATASET=$(parse_long $2); shift; shift;;
        --model)            MODEL=$(parse_long $2); shift; shift;;
        --ratio)            RATIO=$(parse_long $2); shift; shift;;
        --epoch)            EPOCH=$(parse_long $2); shift; shift;;
        --learning_rate)    LR=$(parse_long $2); shift; shift;;
        --batch_size)       BATCH_SIZE=$(parse_long $2); shift; shift;;
        --num_classes)      NB_CLS=$(parse_long $2); shift; shift;;

        -?*) echo "Invalide option $1" >&2; show_help; exit 1;;
        *) break;;
    esac
done
# ___________________________________________________________________________________ #
# ___________________________________________________________________________________ #

folds=("-t 1 2 3 4 5 6 7 8 9 -v 10")
if [ $CROSSVAL -eq 1 ]; then
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
fi

tensorboard_root="../../tensorboard"
checkpoint_root="../../model_save"
tensorboard_path="supervised"
checkpoint_path="supervised"
tensorboard_sufix=""

# ___________________________________________________________________________________ #
parameters=""

# -------- tensorboard and checkpoint path --------
tensorboard_path="${tensorboard_path}/${MODEL}/${RATIO}S"
checkpoint_path="${checkpoint_path}/${MODEL}/${RATIO}S"

parameters="${parameters} --tensorboard_root ${tensorboard_root}"
parameters="${parameters} --checkpoint_root ${checkpoint_root}"

parameters="${parameters} --tensorboard_path ${tensorboard_path}"
parameters="${parameters} --checkpoint_path ${checkpoint_path}"

# -------- dataset ------
parameters="${parameters} --dataset ${DATASET}"

# -------- model --------
parameters="${parameters} --model ${MODEL}"

# -------- training parameters --------
parameters="${parameters} --supervised_ratio ${RATIO}"
parameters="${parameters} --nb_epoch ${EPOCH}"
parameters="${parameters} --learning_rate ${LR}"
parameters="${parameters} --batch_size ${BATCH_SIZE}"
parameters="${parameters} --num_classes ${NB_CLS}"

# -------- resume training --------
if [ $RESUME -eq 1 ]; then
    echo "$RESUME"
    parameters="${parameters} --resume"
fi

run_number=0
for i in ${!folds[*]}
do
    run_number=$(( $run_number + 1 ))

    if [ $CROSSVAL -eq 1 ]; then
        tensorboard_sufix="--tensorboard_sufix run${run_number}"
    else
        tensorboard_sufix=""
    fi
    
    echo python full_supervised.py ${folds[$i]} ${tensorboard_sufix} ${parameters}
    python full_supervised.py ${folds[$i]} ${tensorboard_sufix} ${parameters}
done