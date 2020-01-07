#!/usr/bin/env bash

base_lr=0.01
epsilon=0.1
lambda_cot_max=5
lambda_diff_max=0.25
warmup=160

tensorboard_root="tensorboard/cross_validation/"
fixed_parameters="--tensorboard_dir ${tensorboard_root} --base_lr ${base_lr} --epsilon ${epsilon} --lambda_cot_max ${lambda_cot_max} --lambda_diff_max ${lambda_diff_max} --warm_up ${warmup}"

python co-training.py --job_name run1 ${fixed_parameters} --train_folds 1 2 3 4 5 6 7 8 9 --val_folds 10 --tensorboard_dir ${tensorboard_root}
python co-training.py --job_name run2 ${fixed_parameters} --train_folds 1 2 3 4 5 6 7 8 10 --val_folds 9 --tensorboard_dir ${tensorboard_root}
python co-training.py --job_name run3 ${fixed_parameters} --train_folds 1 2 3 4 5 6 7 9 10 --val_folds 8 --tensorboard_dir ${tensorboard_root}
python co-training.py --job_name run4 ${fixed_parameters} --train_folds 1 2 3 4 5 6 8 9 10 --val_folds 7 --tensorboard_dir ${tensorboard_root}
python co-training.py --job_name run5 ${fixed_parameters} --train_folds 1 2 3 4 5 7 8 9 10 --val_folds 6 --tensorboard_dir ${tensorboard_root}
python co-training.py --job_name run6 ${fixed_parameters} --train_folds 1 2 3 4 6 7 8 9 10 --val_folds 5 --tensorboard_dir ${tensorboard_root}
python co-training.py --job_name run7 ${fixed_parameters} --train_folds 1 2 3 5 6 7 8 9 10 --val_folds 4 --tensorboard_dir ${tensorboard_root}
python co-training.py --job_name run8 ${fixed_parameters} --train_folds 1 2 4 5 6 7 8 9 10 --val_folds 3 --tensorboard_dir ${tensorboard_root}
python co-training.py --job_name run9 ${fixed_parameters} --train_folds 1 3 4 5 6 7 8 9 10 --val_folds 2 --tensorboard_dir ${tensorboard_root}
python co-training.py --job_name run10 ${fixed_parameters} --train_folds 2 3 4 5 6 7 8 9 10 --val_folds 1 --tensorboard_dir ${tensorboard_root}
