#!/bin/sh

#SBATCH --job-name=preprocess
#SBATCH --output=preprocess.out
#SBATCH --error=preprocess.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=48CPUNodes
 
python=/users/samova/lcances/.miniconda3/envs/dl/bin/python
script=../standalone/preprocess_augmentation.py
    
srun ${python} ${script} -a ../dataset/audio -w 32 -A="signal_augmentations.PitchShiftChoice(1.0, choice=(-1.5, -1, 1, 1.5))"
srun ${python} ${script} -a ../dataset/audio -w 32 -A="signal_augmentations.PitchShiftChoice(1.0, choice=(-1.5, -1, 1, 1.5))"
srun ${python} ${script} -a ../dataset/audio -w 32 -A="signal_augmentations.PitchShiftChoice(1.0, choice=(-1.5, -1, 1, 1.5))"
srun ${python} ${script} -a ../dataset/audio -w 32 -A="signal_augmentations.PitchShiftChoice(1.0, choice=(-1.5, -1, 1, 1.5))"

srun ${python} ${script} -a ../dataset/audio -w 32 -A="signal_augmentations.PitchShiftChoice(1.0, choice=(-3, -2, 2, 3))"
srun ${python} ${script} -a ../dataset/audio -w 32 -A="signal_augmentations.PitchShiftChoice(1.0, choice=(-3, -2, 2, 3))"
srun ${python} ${script} -a ../dataset/audio -w 32 -A="signal_augmentations.PitchShiftChoice(1.0, choice=(-3, -2, 2, 3))"
srun ${python} ${script} -a ../dataset/audio -w 32 -A="signal_augmentations.PitchShiftChoice(1.0, choice=(-3, -2, 2, 3))"

srun ${python} ${script} -a ../dataset/audio -w 32 -A="signal_augmentations.Noise(1.0, target_snr=20)"
srun ${python} ${script} -a ../dataset/audio -w 32 -A="signal_augmentations.Noise(1.0, target_snr=20)"
srun ${python} ${script} -a ../dataset/audio -w 32 -A="signal_augmentations.Noise(1.0, target_snr=20)"
srun ${python} ${script} -a ../dataset/audio -w 32 -A="signal_augmentations.Noise(1.0, target_snr=20)"

