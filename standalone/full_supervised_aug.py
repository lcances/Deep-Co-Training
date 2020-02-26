import os

os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NU M_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
import numpy as np
import time
import logging

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.append("../src/")

from datasetManager import DatasetManager
from generators import Dataset
from utils import get_datetime, get_model_from_name, reset_seed, set_logs
from metrics import CategoricalAccuracy
import signal_augmentations
import img_augmentations
import spec_augmentations

# Arguments ========
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--train_folds", nargs="+", required=True, type=int, help="fold to use for training")
parser.add_argument("-v", "--val_folds", nargs="+", required=True, type=int, help="fold to use for validation")
parser.add_argument("--subsampling", default=1.0, type=float, help="subsampling ratio")
parser.add_argument("--subsampling_method", default="balance", type=str, help="subsampling method [random|balance]")
parser.add_argument("--seed", default=1234, type=int, help="Seed for random generation. Use for reproductability")
parser.add_argument("--model", default="cnn", type=str, help="Model to load, see list of model in models.py")
parser.add_argument("-T", "--log_dir", default="Test", required=True, help="Tensorboard working directory")
parser.add_argument("-j", "--job_name", default="default")
parser.add_argument("--log", default="warning", help="Log level")
parser.add_argument("-a","--augments", action="append", help="Augmentation. use as if python script" )
args = parser.parse_args()

# Logging system
set_logs(args.log)

# Reproducibility
reset_seed(args.seed)

# ======== Prep data ========
audio_root = "../dataset/audio"
metadata_root = "../dataset/metadata"
manager = DatasetManager(
    metadata_root, audio_root,
    train_fold=args.train_folds,
    val_fold=args.val_folds,
    subsampling=args.subsampling,
    subsampling_method=args.subsampling_method,
    verbose=1
)

# ---- Prepare augmentation ----
augments = list(map(eval, args.augments))
print(augments)
sys.exit(1)

#  ---- loaders ----
train_dataset = Dataset(manager, train=True, val=False, augments=augments, cached=False)
val_dataset = Dataset(manager, train=False, val=True, cached=True)

# ======== model, loss and optimizer ========
model_func = get_model_from_name(args.model)
m1 = model_func(dataset=manager)
m1.cuda()
logging.info("Model loaded: %s" % model_func.__name__)

criterion_bce = nn.CrossEntropyLoss(reduction="mean")

optimizer = torch.optim.SGD(
    m1.parameters(),
    weight_decay=1e-3,
    lr=0.05
)

# training parameters
nb_epoch = 200
batch_size = 64
nb_batch = len(train_dataset) // batch_size

training_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=20)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# scheduler
lr_lambda = lambda epoch: 0.5 * (np.cos(np.pi * epoch / nb_epoch) + 1)
lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
callbacks = [lr_scheduler]
# callbacks = []

# tensorboard
augmentation_title_part = ""
for augment in augments:
    augmentation_title_part += "_%s" % augment.initial

title = "%s_%s_%s_Cosd-lr_sgd-0.05lr-wd0.001_%de%s" % (
    args.job_name, get_datetime(), model_func.__name__,
    nb_epoch, augmentation_title_part
)
tensorboard = SummaryWriter(log_dir="tensorboard/%s/%s" % (args.log_dir, title), comment=model_func.__name__)


# ======================================================================================================================

#               TRAINING

# ======================================================================================================================
acc_func = CategoricalAccuracy()

for epoch in range(nb_epoch):
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

            #             y_weak_val_pred, _ = model(X_val)
            logits = m1(X_val)

            # calc loss
            weak_loss_val = criterion_bce(logits, y_val)

            # metrics
            #             y_val_pred =torch.log_softmax(logits, dim=1)
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

