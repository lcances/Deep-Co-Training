#!/bin/bash

tensorboard_dir="../../../tensorboard/ubs8k/deep-co-training_independant-loss/1lcm_1ldm"
checkpoint_dir="../../../model_save/ubs8k/deep-co-training_independant-loss/1lcm_1ldm"
warmup_length="1"

# models=("cnn0" "cnn03")
# schedulers=("linear" "sigmoid" "weighted-linear" "weighted-sigmoid")
# learning_rate=("0.003" "0.0005")
# steps=("5" "10")
# epochs=("1000")
# lambda_diff_max=1
# lambda_cot_max=1

models=("cnn03")
schedulers=("weighted-linear")
learning_rate=("0.0005")
steps=("10")
epochs=("1500" "2000" "3000" "4000" "5000")
lambda_diff_max=1
lambda_cot_max=1

for m in ${models[@]}; do
	for s in ${schedulers[@]}; do
		for l in ${learning_rate[@]}; do
			for st in ${steps[@]}; do
				for e in ${epochs[@]}; do
					echo $m $s $l $st
					python co-training_independant_loss.py \
						--tensorboard_path ${tensorboard_dir} \
						--checkpoint_path ${checkpoint_dir} \
						--warmup_length ${warmup_length} \
						--model $m \
						--loss_scheduler $s \
						--learning_rate $l \
						--loss_scheduler_steps $st \
						--nb_epoch $e \
						--lambda_cot_max $lambda_cot_max \
						--lambda_diff_max $lambda_diff_max
				done
						
			done
		done
	done
done
