#!/bin/bash

# ___________________________________________________________________________________ #
function show_help {
    echo "usage:  $BASH_SOURCE [-m MODEL] [-r SUPERVISED RATIO] [-e EPOCH] [-R RESUME] [-l LEARNING_RATE] [-a AUGMENT ID] [-h]"
    echo "    -m MODEL (default cnn03)"
    echo "    -r SUPERVISED RATIO (default 0.1)"
    echo "    -e EPOCH (default 200)"
    echo "    -R RESUME (default FALSE)"
    echo "    -l LEARNING_RATE (default 0.003)"
    echo "    -a AUGMENT ID"
    echo "    -h help"
    
    echo "Available models"
    echo "	cnn0"
    echo "	cnn03"
    echo "	scallable1"

    echo "Available augmentations"
    echo "see augmentation_list.py"
}

# default parameters
MODEL=cnn03
RATIO=0.1
NB_EPOCH=200
LEARNING_RATE=0.003
RESUME=0

while getopts "m:r:e:l:a::R::h" arg; do
  case $arg in
    m) MODEL=$OPTARG;;
    r) RATIO=$OPTARG;;
    e) NB_EPOCH=$OPTARG;;
    l) LEARNING_RATE=$OPTARG;;
    R) RESUME=1;;
    a) AUGMENT+=("$OPTARG");;
    h) show_help;;
    *) 
        echo "invalide option" 1>&2
        show_help
        exit 1
        ;;
  esac
done

echo "nb augmentation " ${#AUGMENT[@]}
# Check augmentation
if [ "${#AUGMENT[@]}" = "1" ]; then
       AUGMENT_1=${AUGMENT[0]}       
       AUGMENT_2=${AUGMENT[0]}
elif [ "${#AUGMENT[@]}" = "2" ]; then
       AUGMENT_1=${AUGMENT[0]}       
       AUGMENT_2=${AUGMENT[1]}
else
	echo "Please provide at one or two augmentations for the adversarial generation"
	show_help
	exit 2
fi

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

tensorboard_path_root="--tensorboard_path ../../tensorboard/ubs8k/deep-co-training_aug4adv/$MODEL/${RATIO}S"
checkpoint_path_root="--checkpoint_path ../../model_save/ubs8k/deep-co-training_aug4adv"

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

# -------- augmentations --------
parameters="${parameters} --augment ${AUGMENT_1}"
parameters="${parameters} --augment ${AUGMENT_2}"

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
    
    echo python co-training-AugAsAdv.py ${folds[$i]} ${tensorboard_sufix} ${parameters}
    # python co-training-AugAsAdv.py ${folds[$i]} ${tensorboard_sufix} ${parameters}
done
