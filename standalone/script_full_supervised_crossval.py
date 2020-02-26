import subprocess
import sys


# Unique augmentation to execute
unique_augments=dict(
    psc1    = "signal_augmentations.PitchShiftChoice(0.5, choice=(-3, -2, 2, 3))",
    psc2    = "signal_augmentations.PitchShiftChoice(0.5, choice=(-1.5, -1, 1, 1.5))",
    # l1      = "signal_augmentations.Level(0.5, rate=(0.9, 1.1))",
    # l2      = "signal_augmentations.Level(0.5, rate=(0.8, 1.2))",
    n1      = "signal_augmentations.Noise(0.5, target_snr=15)",
    n2      = "signal_augmentations.Noise(0.5, target_snr=20)",
    # n3      = "signal_augmentations.Noise(0.5, target_snr=25)",
    rfd01   = "spec_augmentations.RandomFreqDropout(0.5, dropout=0.1)",
    # rfd0075 = "spec_augmentations.RandomFreqDropout(0.5, dropout=0.075)",
    # rfd02   = "spec_augmentations.RandomFreqDropout(0.5, dropout=0.2)",
    # sn25    = "spec_augmentations.Noise(1.0, 25)",
    # rfd005  = "spec_augmentations.RandomFreqDropout(0.5, dropout=0.05)",
)

# split train / val
fold_idx = list(range(1, 11))

train_fold, val_fold = [], []
for i in range(0, 10):
    t_fold = list(map(str, fold_idx[:i] + fold_idx[i+1:]))
    train_fold.append(t_fold)
    val_fold.append(str(fold_idx[i]))


# execution
for key, aug in unique_augments.items():
    for i in range(len(train_fold)):
        print(["python", "full_supervised_aug.py", "-t", *train_fold[i], "-v", val_fold[i], *sys.argv[1:], "-a=\"%s\"" % aug])
        subprocess.call(["python", "full_supervised_aug.py", "-t", *train_fold[i], "-v", val_fold[i], *sys.argv[1:], "-a=%s" % aug])
