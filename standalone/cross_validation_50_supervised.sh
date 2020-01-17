#!/usr/bin/env bash

fixed_parameters="-T tensorboard_50_supervised --base_lr 0.01 --ratio 0.5"

python xx_supervised.py --job_name run1 -t 1 2 3 4 5 6 7 8 9 -v 10 ${fixed_parameters}
python xx_supervised.py --job_name run2 -t 1 2 3 4 5 6 7 8 10 -v 9 ${fixed_parameters}
python xx_supervised.py --job_name run3 -t 1 2 3 4 5 6 7 9 10 -v 8 ${fixed_parameters}
python xx_supervised.py --job_name run4 -t 1 2 3 4 5 6 8 9 10 -v 7 ${fixed_parameters}
python xx_supervised.py --job_name run5 -t 1 2 3 4 5 7 8 9 10 -v 6 ${fixed_parameters}
python xx_supervised.py --job_name run6 -t 1 2 3 4 6 7 8 9 10 -v 5 ${fixed_parameters}
python xx_supervised.py --job_name run7 -t 1 2 3 5 6 7 8 9 10 -v 4 ${fixed_parameters}
python xx_supervised.py --job_name run8 -t 1 2 4 5 6 7 8 9 10 -v 3 ${fixed_parameters}
python xx_supervised.py --job_name run9 -t 1 3 4 5 6 7 8 9 10 -v 2 ${fixed_parameters}
python xx_supervised.py --job_name run10 -t 2 3 4 5 6 7 8 9 10 -v 1 ${fixed_parameters}
