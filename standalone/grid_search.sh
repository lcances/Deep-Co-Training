#!/usr/bin/env bash
#python co-training.py --job_name 1gpu --base_lr 0.05 --lambda_cot_max 10 --lambda_diff_max 0.5 --warm_up 80

# SIMPLE GRID SEARCH ==================================================================================================
# # Change lambda diff max
# python co-training.py --job_name 1gpu --base_lr 0.05 --lambda_cot_max 10 --lambda_diff_max 0.5 --warm_up 80
# python co-training.py --job_name 1gpu --base_lr 0.05 --lambda_cot_max 10 --lambda_diff_max 1.0 --warm_up 80
# python co-training.py --job_name 1gpu --base_lr 0.05 --lambda_cot_max 10 --lambda_diff_max 2.0 --warm_up 80
# python co-training.py --job_name 1gpu --base_lr 0.05 --lambda_cot_max 10 --lambda_diff_max 0.1 --warm_up 80
# python co-training.py --job_name 1gpu --base_lr 0.05 --lambda_cot_max 10 --lambda_diff_max 0.2 --warm_up 80
# python co-training.py --job_name 1gpu --base_lr 0.05 --lambda_cot_max 10 --lambda_diff_max 0.25 --warm_up 80
#
# # change lambda cot max
# python co-training.py --job_name 1gpu --base_lr 0.05 --lambda_cot_max 2 --lambda_diff_max 0.5 --warm_up 80
# python co-training.py --job_name 1gpu --base_lr 0.05 --lambda_cot_max 5 --lambda_diff_max 0.5 --warm_up 80
# python co-training.py --job_name 1gpu --base_lr 0.05 --lambda_cot_max 10 --lambda_diff_max 0.5 --warm_up 80
# python co-training.py --job_name 1gpu --base_lr 0.05 --lambda_cot_max 13 --lambda_diff_max 0.5 --warm_up 80
# python co-training.py --job_name 1gpu --base_lr 0.05 --lambda_cot_max 15 --lambda_diff_max 0.5 --warm_up 80
#
# # change warmup
# python co-training.py --job_name 1gpu --base_lr 0.05 --lambda_cot_max 10 --lambda_diff_max 0.5 --warm_up 40
# python co-training.py --job_name 1gpu --base_lr 0.05 --lambda_cot_max 10 --lambda_diff_max 0.5 --warm_up 60
# python co-training.py --job_name 1gpu --base_lr 0.05 --lambda_cot_max 10 --lambda_diff_max 0.5 --warm_up 80
# python co-training.py --job_name 1gpu --base_lr 0.05 --lambda_cot_max 10 --lambda_diff_max 0.5 --warm_up 100
# python co-training.py --job_name 1gpu --base_lr 0.05 --lambda_cot_max 10 --lambda_diff_max 0.5 --warm_up 120
# python co-training.py --job_name 1gpu --base_lr 0.05 --lambda_cot_max 10 --lambda_diff_max 0.5 --warm_up 140
# python co-training.py --job_name 1gpu --base_lr 0.05 --lambda_cot_max 10 --lambda_diff_max 0.5 --warm_up 160
#
# # Change epsilon
# python co-training.py --job_name 1gpu --base_lr 0.05 --epsilon 0.02
# python co-training.py --job_name 1gpu --base_lr 0.05 --epsilon 0.05
# python co-training.py --job_name 1gpu --base_lr 0.05 --epsilon 0.1
# python co-training.py --job_name 1gpu --base_lr 0.05 --epsilon 0.5
# python co-training.py --job_name 1gpu --base_lr 0.05 --epsilon 1.0
# python co-training.py --job_name 1gpu --base_lr 0.05 --epsilon 2.0
#
# # Change base lr
# python co-training.py --job_name 1gpu --base_lr 0.01
# python co-training.py --job_name 1gpu --base_lr 0.02
# python co-training.py --job_name 1gpu --base_lr 0.03
# python co-training.py --job_name 1gpu --base_lr 0.04
# python co-training.py --job_name 1gpu --base_lr 0.05
# python co-training.py --job_name 1gpu --base_lr 0.06
# python co-training.py --job_name 1gpu --base_lr 0.07
# python co-training.py --job_name 1gpu --base_lr 0.08
# python co-training.py --job_name 1gpu --base_lr 0.09
# python co-training.py --job_name 1gpu --base_lr 0.10

# COMBINE GRID SEARCH ==================================================================================================
# After analysis, I decided to fix few value
# lambda_diff_max = 0.25     warmup = 160     epsilon = 0.1
# I also removed the worst performing value for lambda_cot_max and base_lr.
lambda_cot_max=(2 5 10)
learning_rate=(0.01 0.04 0.06 0.08)

for lambda in ${lambda_cot_max[*]}
do
    for lr in ${learning_rate[*]}
    do
        python co-training.py --tensorboard_dir tensorboard/combination/ --job_name 1gpu --base_lr ${lr} --epsilon 0.1 --lambda_cot_max ${lambda} --lambda_diff_max 0.25 --warm_up 160
    done
done

# After this step, the following parameters should be the best
# base_lr = 0.01
# epsilon = 0.1
# lambda_cot_max = 5
# lambda_diff_max = 0.25
# warmup = 160
