#!/bin/bash

# ___________________________________________________________________________________ #
function show_help {
    echo "usage:  $BASH_SOURCE -m MODEL -r SUPERVISED RATIO"
    echo "    -m MODEL"
    echo "    -r SUPERVISED RATIO"
}

# default parameters
MODEL=cnn0
RATIO=1.0

while getopts ":m:r:" arg; do
  case $arg in
    m) MODEL=$OPTARG;;
    r) RATIO=$OPTARG;;
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

tensorboard_path_root="--tensorboard_path ../../../tensorboard/ubs8k/full_supervised"
checkpoint_path_root="--checkpoint_path ../../../model_save/ubs8k/full_supervised"

# ___________________________________________________________________________________ #
tensorboard_path="${tensorboard_path_root}/${MODEL}/${RATIO}S"
checkpoint_path=${checkpoint_path_root}/${MODEL}/${RATIO}
supervised_ratio="--supervised_ratio ${RATIO}"
model="--model ${MODEL}"

run_number=1
for i in ${!folds[*]}
do
    run_number=$(( $run_number + 1 ))
    tensorboard_sufix="--tensorboard_sufix run${run_number}"
    
    echo python full_supervised.py ${folds[$i]} ${model} ${tensorboard_path} ${checkpoint_path} ${supervised_ratio} ${tensorboard_sufix}
    python full_supervised.py ${folds[$i]} ${model} ${tensorboard_path} ${checkpoint_path} ${supervised_ratio} ${tensorboard_sufix}
done