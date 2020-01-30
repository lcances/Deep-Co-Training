#!/usr/bin/env bash

fixed_parameters="-T tensorboard_supervised_aug_scallable2"

python full_supervised_aug_scallable2.py -t 1 2 3 4 5 6 7 8 9 -v 10 ${fixed_parameters}
python full_supervised_aug_scallable2.py -t 1 2 3 4 5 6 7 8 10 -v 9 ${fixed_parameters}
python full_supervised_aug_scallable2.py -t 1 2 3 4 5 6 7 9 10 -v 8 ${fixed_parameters}
python full_supervised_aug_scallable2.py -t 1 2 3 4 5 6 8 9 10 -v 7 ${fixed_parameters}
python full_supervised_aug_scallable2.py -t 1 2 3 4 5 7 8 9 10 -v 6 ${fixed_parameters}
python full_supervised_aug_scallable2.py -t 1 2 3 4 6 7 8 9 10 -v 5 ${fixed_parameters}
python full_supervised_aug_scallable2.py -t 1 2 3 5 6 7 8 9 10 -v 4 ${fixed_parameters}
python full_supervised_aug_scallable2.py -t 1 2 4 5 6 7 8 9 10 -v 3 ${fixed_parameters}
python full_supervised_aug_scallable2.py -t 1 3 4 5 6 7 8 9 10 -v 2 ${fixed_parameters}
python full_supervised_aug_scallable2.py  -t 2 3 4 5 6 7 8 9 10 -v 1 ${fixed_parameters}
