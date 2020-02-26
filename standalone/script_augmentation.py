import subprocess
import sys


# Unique augmentation to execute
unique_augments=dict(
    psc1    = "signal_augmentations.PitchShiftChoice(0.5, choice=(-3, -2, 2, 3))",
    psc2    = "signal_augmentations.PitchShiftChoice(0.5, choice=(-1.5, -1, 1, 1.5))",
    l1      = "signal_augmentations.Level(0.5, rate=(0.9, 1.1))",
    l2      = "signal_augmentations.Level(0.5, rate=(0.8, 1.2))",
    n1      = "signal_augmentations.Noise(0.5, target_snr=15)",
    n2      = "signal_augmentations.Noise(0.5, target_snr=20)",
    n3      = "signal_augmentations.Noise(0.5, target_snr=25)",
    rfd01   = "spec_augmentations.RandomFreqDropout(0.5, dropout=0.1)",
    rfd0075 = "spec_augmentations.RandomFreqDropout(0.5, dropout=0.075)",
    rfd02   = "spec_augmentations.RandomFreqDropout(0.5, dropout=0.2)",
    sn25    = "spec_augmentations.Noise(1.0, 25)",
    rfd005  = "spec_augmentations.RandomFreqDropout(0.5, dropout=0.05)",
)


# execution
print(sys.argv)
for key, aug in unique_augments.items():
    subprocess.call(["python", "full_supervised_aug.py", *sys.argv[1:], "-a", "\"%s\"" % aug])
