#!/bin/bash

# ___________________________________________________________________________________ #
function show_help {
    echo "usage:  $BASH_SOURCE [-D DATASET] [-m MODEL] [-r SUPERVISED RATIO] [-e EPOCH] [-R RESUME] [-s LAMBDA_SUP_MAX] [-c LAMBDA_COT_MAX] [-d LAMBDA_DIFF_MAX] [-h]"
    echo "    -D DATASET (default ubs8k)"
    echo "    -m MODEL (default cnn03)"
    echo "    -r SUPERVISED RATIO (default 0.1)"
    echo "    -e EPOCH (default 200)"
    echo "    -R RESUME (default FALSE)"
    echo "    -s LAMBDA SUP MAX (default 1)"
    echo "    -c LAMBDA COT MAX (default 10)"
    echo "    -d LAMBDA_DIFF_MAX (default 0.5)"
    echo "    -h help"
    
    echo "Available datasets"
    echo "    ubs8k"
    echo "    cifar10"
    
    echo "Available models"
    echo "\t cnn0"
    echo "\t cnn03"
    echo "\t scallable1"
}

# default parameters
DATASET="ubs8k"
MODEL=cnn03
RATIO=0.1
NB_EPOCH=200
RESUME=0
LSM=1
LCM=10
LDM=0.5
SCRIPT="co-training.py"

while getopts "D:m:r:e:s::R::h" arg; do
  case $arg in
    D) DATASET=$OPTARG;;
    m) MODEL=$OPTARG;;
    r) RATIO=$OPTARG;;
    e) NB_EPOCH=$OPTARG;;
    s) LSM=$OPTARG;;
    c) LCM=$OPTARG;;
    d) LDM=$OPTARG;;
    R) RESUME=1;;
    h) show_help;;
    *) 
        echo "invalide option" 1>&2
        show_help
        exit 1
        ;;
  esac
done

# ___________________________________________________________________________________ #
# ___________________________________________________________________________________ #

folds=(
	"-t 2 3 4 5 6 7 8 9 10 -v 1" \
	# "-t 1 3 4 5 6 7 8 9 10 -v 2" \
	# "-t 1 2 4 5 6 7 8 9 10 -v 3" \
	# "-t 1 2 3 5 6 7 8 9 10 -v 4" \
	# "-t 1 2 3 4 6 7 8 9 10 -v 5" \
	# "-t 1 2 3 4 5 7 8 9 10 -v 6" \
	# "-t 1 2 3 4 5 6 8 9 10 -v 7" \
	# "-t 1 2 3 4 5 6 7 9 10 -v 8" \
	# "-t 1 2 3 4 5 6 7 8 10 -v 9" \
	# "-t 1 2 3 4 5 6 7 8 9 -v 10" \
)

tensorboard_path_root="--tensorboard_path ../../tensorboard/${DATASET}/deep-co-training"
checkpoint_path_root="--checkpoint_path ../../model_save/${DATASET}/deep-co-training"

# ___________________________________________________________________________________ #
parameters=""

# -------- tensorboard and checkpoint path --------
tensorboard_path="${tensorboard_path_root}/${MODEL}/${RATIO}S"
checkpoint_path=${checkpoint_path_root}/${MODEL}/${RATIO}
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
