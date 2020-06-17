#!/usr/bin/env bash

# # SIMPLE GRID SEARCH ==================================================================================================
fixed_parameters="-t 1 2 3 4 5 6 7 8 9 -v 10 -T grid_search --job_name cnn_10r --subsampling 1.0 --subsampling_method balance"
#
# # Change lambda diff max
# python co-training.py ${fixed_parameters} --lambda_diff_max 0.5 -T grid_search/lambda_diff_max
# python co-training.py ${fixed_parameters} --lambda_diff_max 1.0 -T grid_search/lambda_diff_max
# python co-training.py ${fixed_parameters} --lambda_diff_max 2.0 -T grid_search/lambda_diff_max
# python co-training.py ${fixed_parameters} --lambda_diff_max 0.1 -T grid_search/lambda_diff_max
# python co-training.py ${fixed_parameters} --lambda_diff_max 0.2 -T grid_search/lambda_diff_max
# python co-training.py ${fixed_parameters} --lambda_diff_max 0.25 -T grid_search/lambda_diff_max
#
# # change lambda cot max
# python co-training.py ${fixed_parameters} --lambda_cot_max 2 -T grid_search/lambda_cot_max
# python co-training.py ${fixed_parameters} --lambda_cot_max 5 -T grid_search/lambda_cot_max
# python co-training.py ${fixed_parameters} --lambda_cot_max 10 -T grid_search/lambda_cot_max
# python co-training.py ${fixed_parameters} --lambda_cot_max 13 -T grid_search/lambda_cot_max
# python co-training.py ${fixed_parameters} --lambda_cot_max 15 -T grid_search/lambda_cot_max
#
# # change warmup
# python co-training.py ${fixed_parameters} --warm_up 40 -T grid_search/warmup
# python co-training.py ${fixed_parameters} --warm_up 60 -T grid_search/warmup
# python co-training.py ${fixed_parameters} --warm_up 80 -T grid_search/warmup
# python co-training.py ${fixed_parameters} --warm_up 100 -T grid_search/warmup
# python co-training.py ${fixed_parameters} --warm_up 120 -T grid_search/warmup
# python co-training.py ${fixed_parameters} --warm_up 140 -T grid_search/warmup
# python co-training.py ${fixed_parameters} --warm_up 160 -T grid_search/warmup
#
# # Change epsilon
# python co-training.py ${fixed_parameters} --epsilon 0.02 -T grid_search/epsilon
# python co-training.py ${fixed_parameters} --epsilon 0.05 -T grid_search/epsilon
# python co-training.py ${fixed_parameters} --epsilon 0.1 -T grid_search/epsilon
# python co-training.py ${fixed_parameters} --epsilon 0.5 -T grid_search/epsilon
# python co-training.py ${fixed_parameters} --epsilon 1.0 -T grid_search/epsilon
# python co-training.py ${fixed_parameters} --epsilon 2.0 -T grid_search/epsilon
#
# # Change base lr
# python co-training.py ${fixed_parameters} --base_lr 0.01 -T grid_search/learning_rate
# python co-training.py ${fixed_parameters} --base_lr 0.02 -T grid_search/learning_rate
# python co-training.py ${fixed_parameters} --base_lr 0.03 -T grid_search/learning_rate
# python co-training.py ${fixed_parameters} --base_lr 0.04 -T grid_search/learning_rate
# python co-training.py ${fixed_parameters} --base_lr 0.05 -T grid_search/learning_rate
# python co-training.py ${fixed_parameters} --base_lr 0.06 -T grid_search/learning_rate
# python co-training.py ${fixed_parameters} --base_lr 0.07 -T grid_search/learning_rate
# python co-training.py ${fixed_parameters} --base_lr 0.08 -T grid_search/learning_rate
# python co-training.py ${fixed_parameters} --base_lr 0.09 -T grid_search/learning_rate
# python co-training.py ${fixed_parameters} --base_lr 0.10 -T grid_search/learning_rate

# change weight decay
fixed_parameters="-t 1 2 3 4 5 6 7 8 9 -v 10 --base_lr 0.01 --epsilon 0.1 --lambda_cot_max 5 --lambda_diff_max 0.25 --warm_up 160 --subsampling 0.1 --subsampling_method balance --epochs 200"
python co-training.py ${fixed_parameters} --decay 0.1 -T grid_search/decay
python co-training.py ${fixed_parameters} --decay 0.01 -T grid_search/decay
python co-training.py ${fixed_parameters} --decay 0.005 -T grid_search/decay
python co-training.py ${fixed_parameters} --decay 0.001 -T grid_search/decay
python co-training.py ${fixed_parameters} --decay 0.0005 -T grid_search/decay
python co-training.py ${fixed_parameters} --decay 0.0001 -T grid_search/decay
python co-training.py ${fixed_parameters} --decay 0.00001 -T grid_search/decay

# COMBINE GRID SEARCH ==================================================================================================
# After analysis, I decided to fix few value
# lambda_diff_max = 0.25     warmup = 160     epsilon = 0.1
# I also removed the worst performing value for lambda_cot_max and base_lr.
# lambda_cot_max=(2 5 10)
# learning_rate=(0.01 0.04 0.06 0.08)
#
# for lambda in ${lambda_cot_max[*]}
# do
#     for lr in ${learning_rate[*]}
#     do
#         python co-training.py --tensorboard_dir tensorboard/combination/ --job_name 1gpu --base_lr ${lr} --epsilon 0.1 --lambda_cot_max ${lambda} --lambda_diff_max 0.25 --warm_up 160
#     done
# done

# After this step, the following parameters should be the best
# base_lr = 0.01
# epsilon = 0.1
# lambda_cot_max = 5
# lambda_diff_max = 0.25
# warmup = 160
