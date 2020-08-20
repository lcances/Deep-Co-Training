#!/bin/bash


models=("cnn03")
schedulers=("linear")
learning_rate=("0.0005")
steps=("10")
epochs=("1000")
lambda_diff_max=(1)
lambda_cot_max=(1)
warmup_length=(1)
cycle=(1)
beta=(1)
plsup_mini=("0.0" "0.05" "0.1" "0.2" "0.3" "0.4" "0.5")

tensorboard_dir="../../tensorboard/deep-co-training_independant-loss/${lambda_cot_max}lcm_${lambda_diff_max}ldm/grid_search"
checkpoint_dir="../../model_save/deep-co-training_independant-loss/${lambda_cot_max}lcm_${lambda_diff_max}ldm/grid_search"

for m in ${models[@]}; do
for s in ${schedulers[@]}; do
for l in ${learning_rate[@]}; do
for st in ${steps[@]}; do
for e in ${epochs[@]}; do
for lcm in ${lambda_cot_max[@]}; do
for ldm in ${lambda_diff_max[@]}; do
for lc in ${cycle[@]}; do
for lb in ${beta[@]}; do
for psm in ${plsup_mini[@]}; do
	tensorboard_sufix="${lc}cycle_${lb}beta_${psm}m-plsup"
	python co-training_independant_loss.py \
		--tensorboard_path ${tensorboard_dir} \
		--checkpoint_path ${checkpoint_dir} \
		--tensorboard_sufix ${tensorboard_sufix} \
		--warmup_length ${warmup_length} \
		--model $m \
		--learning_rate $l \
		--nb_epoch $e \
		--lambda_cot_max $lcm \
		--lambda_diff_max $ldm \
		--loss_scheduler $s \
		--steps $st \
		--cycle $lc \
		--beta $lb \
		--plsup_mini $psm
done			
done
done
done			
done			
done			
done
done
done
done
