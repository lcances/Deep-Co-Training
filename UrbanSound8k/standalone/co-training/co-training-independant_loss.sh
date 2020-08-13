#!/bin/bash

# ___________________________________________________________________________________ #
function show_help {
    echo "usage:  $BASH_SOURCE [-m MODEL] [-r SUPERVISED RATIO] [-e EPOCH] [-R RESUME] [-s LOSS SCHEDULER] [-l LEARNING_RATE] [-S STEPS] [-c LAMBDA_COT_MAX] [-d LAMBDA DIFF MAX] [-t LOG SUB DIR] [-h]"
    echo "    -m MODEL (default cnn03)"
    echo "    -r SUPERVISED RATIO (default 0.1)"
    echo "    -e EPOCH (default 3000)"
    echo "    -R RESUME (default FALSE)"
    echo "    -s LOSS SCHEDULER (default weighted-linear)"
    echo "    -S STEPS (default 10)"
    echo "    -l LEARNING_RATE (default 0.0005)"
    echo "    -c LAMBDA COT MAX (default 10)"
    echo "    -d LAMBDA DIFF MAX (default 0.5)"
    echo "    -t LOG SUB DIRECTORY (default \"\")"
    echo "    -h help"
    
    echo "Available models"
    echo "	cnn0"
    echo "	cnn03"
    echo "	scallable1"

    echo "Available loss scheduler"
    echo "	linear"
    echo "	weighted linear"
    echo "	sigmoid"
    echo "	weighted sigmoid"
}

# default parameters
MODEL=cnn03
RATIO=0.1
NB_EPOCH=3000
LOSS_SCHEDULER="weighted-linear"
STEPS=10
LEARNING_RATE=0.0005
LAMBDA_COT_MAX=10
LAMBDA_DIFF_MAX=0.5
RESUME=0
LOG_SUB_DIR=""

while getopts "m:r:e:s:S:l:c:d:t::R::h" arg; do
  case $arg in
    m) MODEL=$OPTARG;;
    r) RATIO=$OPTARG;;
    e) NB_EPOCH=$OPTARG;;
    s) LOSS_SCHEDULER=$OPTARG;;
    S) STEPS=$OPTARG;;
    l) LEARNING_RATE=$OPTARG;;
    c) LAMBDA_COT_MAX=$OPTARG;;
    d) LAMBDA_DIFF_MAX=$OPTARG;;
    t) LOG_SUB_DIR=$OPTARG;;
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

tensorboard_path_root="--tensorboard_path ../../../tensorboard/ubs8k/deep-co-training_independant-loss/${LOG_SUB_DIR}"
checkpoint_path_root="--checkpoint_path ../../../model_save/ubs8k/deep-co-training_independant-loss/${LOG_SUB_DIR}"

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
parameters="${parameters} --learning_rate ${LEARNING_RATE}"

parameters="${parameters} --lambda_cot_max ${LAMBDA_COT_MAX}"
parameters="${parameters} --lambda_diff_max ${LAMBDA_DIFF_MAX}"

parameters="${parameters} --loss_scheduler ${LOSS_SCHEDULER}"
parameters="${parameters} --loss_scheduler_steps ${STEPS}"

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
    
    echo python co-training_independant_loss.py ${folds[$i]} ${tensorboard_sufix} ${parameters}
    python co-training_independant_loss.py ${folds[$i]} ${tensorboard_sufix} ${parameters}
done
