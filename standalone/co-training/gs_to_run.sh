#!/bin/bash

dataset="ubs8k"
models=("cnn03")
schedulers=("annealing-cosine" "weighted-annealing-cosine")
learning_rate=("0.0005")
steps=("10" "5")
epochs=("1000")
lambda_diff_max=(1)
lambda_cot_max=(1)
warmup_length=(1)
cycle=(8 12 16)
beta=(1 2 3)
plsup_mini=("0.0")

tensorboard_dir="../../tensorboard/${dataset}/deep-co-training_independant-loss/${lambda_cot_max}lcm_${lambda_diff_max}ldm/grid_search"
checkpoint_dir="../../model_save/${dataset}/deep-co-training_independant-loss/${lambda_cot_max}lcm_${lambda_diff_max}ldm/grid_search"

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
	tensorboard_sufix="${lc}cycle_${lb}beta_${psm}m_plsup"
	python co-training_independant_loss.py \
        --dataset ${dataset} \
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
