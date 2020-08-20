#!/usr/bin/env bash

fixed_parameters="-t 1 2 3 4 5 6 7 8 9 -v 10 --subsampling 0.1 --subsampling_method balance"

declare -a augments=(
    "signal_augmentations.PitchShiftChoice(0.5, choice=(-3, -2, 2, 3))"
    "signal_augmentations.PitchShiftChoice(0.5, choice=(-1.5, -1, 1, 1.5))"
    "signal_augmentations.Level(0.5, rate=(0.9, 1.1)"
    "signal_augmentations.Level(0.5, rate=(0.8, 1.2)"
    "signal_augmentations.Noise(0.5, target_snr=15)"
    "signal_augmentations.Noise(0.5, target_snr=20)"
    "signal_augmentations.Noise(0.5, target_snr=25)"
)

declare -a jobnames=(
    "PSC1"
    "PSC2"
    "L1"
    "L2"
    "N1"
    "N2"
    "N3"
)

# execution
for i in ${!augments[@]}; do
    python full_supervised_aug.py ${fixed_parameters} --job_name ${jobnames[$i]} -a ${augments[$i]}
done
