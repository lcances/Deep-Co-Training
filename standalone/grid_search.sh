#!/usr/bin/env bash
# python co-training.py --job_name 1gpu --base_lr 0.05 --lambda_cot_max 10 --lambda_diff_max 0.5 --warm_up 80
#
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

# Change epsilon
python co-training.py --job_name 1gpu --base_lr 0.05 --epsilon 0.02
python co-training.py --job_name 1gpu --base_lr 0.05 --epsilon 0.05
python co-training.py --job_name 1gpu --base_lr 0.05 --epsilon 0.1
python co-training.py --job_name 1gpu --base_lr 0.05 --epsilon 0.5
python co-training.py --job_name 1gpu --base_lr 0.05 --epsilon 1.0
python co-training.py --job_name 1gpu --base_lr 0.05 --epsilon 2.0
