#!/bin/bash

source ../bash_scripts/parse_option.sh

function show_help {
	echo "usage: $BASH_SOURCE --dataset --model"
	echo ""
	echo "--dataset   DATASET [\"ubs8k\" | \"esc10\" | \"esc50\" | \"speechcommand\"]"
	echo "--model     MODEL"
	echo ""
}

# -------- default argument value --------
DATASET="speechcommand"
MODEL="wideresnet28_2"

# -------- argument parser --------
while :; do
	if ! [ "$1" ]; then break; fi

	case $1 in
		-h) show_help;;

		--dataset)  DATASET=$(parse_long $2); shift; shift;;
		--model)    MODEL=$(parse_long $2); shift; shift;;

		-?*) echo "WARN/ unknown option" $1 >&2;;
	esac
done

# -------- execution --------
# ---- google speech command
# no predefined cross validation for GSC, so we are doing multiple run
train_folds=null
val_folds=null
seed=1234,1235,1236,1237,1238

# ---- UrbandSound8K 
if [ "$DATASET" = "ubs8k" ]; then
	declare -a train_folds=("[1,2,3,4,5,6,7,8,9]"
		"[2,3,4,5,6,7,8,9,10]"
		"[3,4,5,6,7,8,9,10,1]"
		"[4,5,6,7,8,9,10,1,2]"
		"[5,6,7,8,9,10,1,2,3]"
		"[6,7,8,9,10,1,2,3,4]"
		"[7,8,9,10,1,2,3,4,5]"
		"[8,9,10,1,2,3,4,5,6]"
		"[9,10,1,2,3,4,5,6,7]"
		"[10,1,2,3,4,5,6,7,8]")
	val_folds=("[10]" "[1]" "[2]" "[3]" "[4]" "[5]" "[6]" "[7]" "[8]" "[9]")
	seed=1234
fi

# ---- ESC-10
if [ "$DATASET" = "esc10" ] || [ "$DATASET" = "esc50" ]; then
	declare -a train_folds=("[1,2,3,4]"
		"[2,3,4,5]"
		"[3,4,5,1]"
		"[4,5,1,2]"
		"[5,1,2,3]")
	val_folds=("[5]" "[1]" "[2]" "[3]" "[4]")
	seed=1234
fi

# Use hydra to handle the multiple run automatically
# only for gsc unfortunately :'(
if [ "$DATASET" = "speechcommand" ]; then
	python supervised.py -m -cn ../../config/deep-co-training/deep-co-training.yaml \
		dataset.dataset=$DATASET \
		model.model=$MODEL \
		train_param.seed=$seed
fi

# manually executing each run for esc and ubs8k
if [ "$DATASET" = "esc10" ] || [ "$DATASET" = "esc50" ] || [ "$DATASET" = "ubs8k" ]; then
	for i in ${!train_folds[@]}; do
	       python supervised.py -cn ../../config/deep-co-training/deep-co-training.yaml \
	       	dataset.dataset=$DATASET \
 		model.model=$MODEL \
		train_param.train_folds=${train_folds[$i]} \
		train_param.val_folds=${val_folds[$i]} \
		train_param.seed=$seed
	done
fi	




