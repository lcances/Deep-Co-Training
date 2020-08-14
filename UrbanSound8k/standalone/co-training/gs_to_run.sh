#!/bin/bash


models=("cnn03")
schedulers=("annealing-cosine" "weighted-annealing-cosine")
learning_rate=("0.0005")
steps=("10" "5000")
epochs=("1000")
lambda_diff_max=(1)
lambda_cot_max=(1)
warmup_length=(1)
cycle=(8 12 16)
beta=(1 2 3)

tensorboard_dir="../../tensorboard/deep-co-training_independant-loss/${lambda_cot_max}lcm_${lambda_diff_max}ldm"
checkpoint_dir="../../model_save/deep-co-training_independant-loss/${lambda_cot_max}lcm_${lambda_diff_max}ldm"

for m in ${models[@]}; do
for s in ${schedulers[@]}; do
for l in ${learning_rate[@]}; do
for st in ${steps[@]}; do
for e in ${epochs[@]}; do
for lcm in ${lambda_cot_max[@]}; do
for ldm in ${lambda_diff_max[@]}; do
for lc in ${cycle[@]}; do
for lb in ${beta[@]}; do
	echo $m $s $l $st
	python co-training_independant_loss.py \
		--tensorboard_path ${tensorboard_dir} \
		--checkpoint_path ${checkpoint_dir} \
		--warmup_length ${warmup_length} \
		--model $m \
		--learning_rate $l \
		--nb_epoch $e \
		--lambda_cot_max $lcm \
		--lambda_diff_max $ldm \
		--loss_scheduler $s \
		--loss_scheduler_steps $st \
		--loss_scheduler_cycle $lc \
		--loss_scheduler_beta $lb
		
done			
done			
done			
done			
done			
done
done
done
done
