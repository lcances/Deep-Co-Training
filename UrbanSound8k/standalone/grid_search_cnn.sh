#!/usr/bin/env bash
script=co-training_cnn.py
#python ${script} --job_name 1gpu --base_lr 0.05 --lambda_cot_max 10 --lambda_diff_max 0.5 --warm_up 80

# SIMPLE GRID SEARCH ==================================================================================================
# # Change lambda diff max
# python ${script} --job_name cnn_gs --lambda_diff_max 0.5 
# python ${script} --job_name cnn_gs --lambda_diff_max 1.0 
# python ${script} --job_name cnn_gs --lambda_diff_max 2.0 
# python ${script} --job_name cnn_gs --lambda_diff_max 0.1 
# python ${script} --job_name cnn_gs --lambda_diff_max 0.2 
# python ${script} --job_name cnn_gs --lambda_diff_max 0.25 
#
# # change lambda cot max
# python ${script} --job_name cnn_gs --lambda_cot_max 2 
# python ${script} --job_name cnn_gs --lambda_cot_max 5 
# python ${script} --job_name cnn_gs --lambda_cot_max 10
# python ${script} --job_name cnn_gs --lambda_cot_max 13
# python ${script} --job_name cnn_gs --lambda_cot_max 15
#
# # change warmup
# python ${script} --job_name cnn_gs --warm_up 40
# python ${script} --job_name cnn_gs --warm_up 60
# python ${script} --job_name cnn_gs --warm_up 80
# python ${script} --job_name cnn_gs --warm_up 100
# python ${script} --job_name cnn_gs --warm_up 120
# python ${script} --job_name cnn_gs --warm_up 140
# python ${script} --job_name cnn_gs --warm_up 160
#
# # Change epsilon
# python ${script} --job_name cnn_gs --epsilon 0.02
# python ${script} --job_name cnn_gs --epsilon 0.05
# python ${script} --job_name cnn_gs --epsilon 0.1
# python ${script} --job_name cnn_gs --epsilon 0.5
# python ${script} --job_name cnn_gs --epsilon 1.0
# python ${script} --job_name cnn_gs --epsilon 2.0
#
# # Change base lr
# python ${script} --job_name cnn_gs --base_lr 0.01
# python ${script} --job_name cnn_gs --base_lr 0.02
# python ${script} --job_name cnn_gs --base_lr 0.03
# python ${script} --job_name cnn_gs --base_lr 0.04
# python ${script} --job_name cnn_gs --base_lr 0.05
# python ${script} --job_name cnn_gs --base_lr 0.06
# python ${script} --job_name cnn_gs --base_lr 0.07
# python ${script} --job_name cnn_gs --base_lr 0.08
# python ${script} --job_name cnn_gs --base_lr 0.09
# python ${script} --job_name cnn_gs --base_lr 0.10


# COMBINE GRID SEARCH ==================================================================================================
# After analysis, I decided to fix few value
# lambda_diff_max = 0.25     warmup = 160     epsilon = 0.1
# I also removed the worst performing value for lambda_cot_max and base_lr.
lambda_cot_max=(2 5 10)
learning_rate=(0.01 0.04 0.06 0.08)

# for lambda in ${lambda_cot_max[*]}
# do
#     for lr in ${learning_rate[*]}
#     do
#         python ${script} --tensorboard_dir tensorboard/combination/ --job_name cnn_gs --base_lr ${lr} --epsilon 0.1 --lambda_cot_max ${lambda} --lambda_diff_max 0.25 --warm_up 160
#     done
# done

# After this step, the following parameters should be the best
# base_lr = 0.01
# epsilon = 0.1
# lambda_cot_max = 5
# lambda_diff_max = 0.25
# warmup = 160

# Change weight decay
fixed_parameters="--job_name cnn_gs --base_lr 0.01 --epsilon 0.1 --lambda_cot_max 5 --lambda_diff_max 0.25 --warmup 160"
python ${script} ${fixed_parameters} --decay 0.1
python ${script} ${fixed_parameters} --decay 0.01
python ${script} ${fixed_parameters} --decay 0.005
python ${script} ${fixed_parameters} --decay 0.001
python ${script} ${fixed_parameters} --decay 0.0005
python ${script} ${fixed_parameters} --decay 0.0001
