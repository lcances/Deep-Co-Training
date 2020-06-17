import subprocess
import sys

# split train / val
fold_idx = list(range(1, 11))

train_fold, val_fold = [], []
for i in range(0, 10):
    t_fold = list(map(str, fold_idx[:i] + fold_idx[i+1:]))
    train_fold.append(t_fold)
    val_fold.append(str(fold_idx[i]))


# execution
for i in range(len(train_fold)):
    print(["python", "full_supervised_aug.py", "-t", *train_fold[i], "-v", val_fold[i], *sys.argv[1:]])
    subprocess.call(["python", "full_supervised_aug.py", "--job_name", "run%s" % i, "-t", *train_fold[i], "-v", val_fold[i], *sys.argv[1:]])
