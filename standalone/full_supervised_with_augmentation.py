import os

os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NU M_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
import numpy as np
import tqdm
import time

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.append("../src/")

from datasetManager import DatasetManager
from generators import Generator
import models
from utils import get_datetime
from metrics import CategoricalAccuracy
import signal_augmentations as sa

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", default="cnn", type=str, help="The name of the model to load")
parser.add_argument("-t", "--train", nargs="+", required=True, type=int, help="fold to use for training")
parser.add_argument("-v", "--val", nargs="+", required=True, type=int, help="fold to use for validation")
parser.add_argument("-T", "--log_dir", required=True, help="Tensorboard working directory")
args = parser.parse_args()


def reset_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
reset_seed()

# Prep data
audio_root = "../dataset/audio"
metadata_root = "../dataset/metadata"
print(args.train)
dataset = DatasetManager(
    metadata_root=metadata_root,
    audio_root=audio_root,
    train_fold=args.train,
    val_fold=args.val,
    verbose=1
)

# prep model
torch.cuda.empty_cache()

def get_model_from_name(model_name):
    import inspect

    for name, obj in inspect.getmembers(models):
        if inspect.isclass(obj):
            if obj.__name__ == model_name:
                return obj


model_func = get_model_from_name(args.model)

m1 = model_func()
m1.cuda()

# loss and optimizer
criterion_bce = nn.CrossEntropyLoss(reduction="mean")

optimizer = torch.optim.SGD(
    m1.parameters(),
    weight_decay=1e-3,
    lr=0.05
)

# train and val loaders
ps1 = sa.PitchShiftChoice(0.5, choice=(-2, -1, 1, 2))
ps2 = sa.PitchShiftChoice(0.5, choice=(-3.5, -2.5, 2.5, 3.5))
augments = [ps1, ps2]
train_dataset = Generator(dataset, augments=augments)

x, y = train_dataset.validation
x = torch.from_numpy(x)
y = torch.from_numpy(y)
val_dataset = torch.utils.data.TensorDataset(x, y)

# training parameters
nb_epoch = 100
batch_size = 32
nb_batch = len(train_dataset) // batch_size

training_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=5)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# scheduler
lr_lambda = lambda epoch: 0.5 * (np.cos(np.pi * epoch / nb_epoch) + 1)
lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
callbacks = [lr_scheduler]
# callbacks = []

# tensorboard
title = "%s_cnn_Cosd-lr_sgd-0.01lr-wd0.001_%de_0.5n" % (get_datetime(), nb_epoch)
tensorboard = SummaryWriter(log_dir="%s/%s" % (args.log_dir, title), comment=model_func.__name__)


# ======================================================================================================================

#               TRAINING

# ======================================================================================================================
acc_func = CategoricalAccuracy()

for epoch in tqdm.tqdm_notebook(range(nb_epoch)):
    start_time = time.time()
    print("")

    acc_func.reset()

    m1.train()

    for i, (X, y) in enumerate(training_loader):
        # Transfer to GPU
        X = X.cuda().float()
        y = y.cuda().long()

        # predict
        logits = m1(X)

        loss = criterion_bce(logits, y)

        # calc metrics
        _, y_pred = torch.max(logits, 1)
        acc = acc_func(y_pred, y)

        # ======== back propagation ========
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ======== history ========
        print("Epoch {}, {:d}% \t ce: {:.4f} - acc: {:.4f} - took: {:.2f}s".format(
            epoch + 1,
            int(100 * (i + 1) / nb_batch),
            loss.item(),
            acc,
            time.time() - start_time
        ), end="\r")

    # using tensorboard to monitor loss and acc
    tensorboard.add_scalar('train/ce', loss.item(), epoch)
    tensorboard.add_scalar("train/acc", 100. * acc, epoch)

    # Validation
    with torch.set_grad_enabled(False):
        # reset metrics
        acc_func.reset()
        m1.eval()

        for X_val, y_val in val_loader:
            # Transfer to GPU
            X_val = X_val.cuda().float()
            y_val = y_val.cuda().long()

            logits = m1(X_val)

            # calc loss
            weak_loss_val = criterion_bce(logits, y_val)

            # metrics
            _, y_val_pred = torch.max(logits, 1)
            acc_val = acc_func(y_val_pred, y_val)

            # Print statistics
            print(
                "Epoch {}, {:d}% \t ce: {:.4f} - acc: {:.4f} - ce val: {:.4f} - acc val: {:.4f} - took: {:.2f}s".format(
                    epoch + 1,
                    int(100 * (i + 1) / nb_batch),
                    loss.item(),
                    acc,
                    weak_loss_val.item(),
                    acc_val,
                    time.time() - start_time
                ), end="\r")

        # using tensorboard to monitor loss and acc
        tensorboard.add_scalar('validation/ce', weak_loss_val.item(), epoch)
        tensorboard.add_scalar("validation/acc", 100. * acc_val, epoch)

    for callback in callbacks:
        callback.step()
