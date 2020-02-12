import os

os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NU M_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
import numpy as np
import tqdm
import random
import time

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.append("../src/")

from datasetManager import DatasetManager
from generators import Dataset
from utils import get_datetime, get_model_from_name
from metrics import CategoricalAccuracy

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--train", nargs="+", required=True, type=int, help="fold to use for training")
parser.add_argument("-v", "--val", nargs="+", required=True, type=int, help="fold to use for validation")
parser.add_argument("-s", "--subsampling", default=1.0, type=float, help="subsampling ratio")
parser.add_argument("-sm", "--subsampling_method", default="balance", type=str, help="subsampling method [random|balance]")
parser.add_argument('--seed', default=1234, type=int)
parser.add_argument("--base_lr", default=0.05, type=float, help="initiation learning rate to train model")
parser.add_argument("--decay", default=0.001, type=float, help="L2 regularization")
parser.add_argument("-T", "--log_dir", default="Test", required=True, help="Tensorboard working directory")
parser.add_argument("--model", default="cnn", type=str, help="model to load")
args = parser.parse_args()


# ## Reproducibility
def reset_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark = False
reset_seed(args.seed)


# Prep data
audio_root = "../dataset/audio"
metadata_root = "../dataset/metadata"
print(args.train)
dataset = DatasetManager(
    metadata_root=metadata_root,
    audio_root=audio_root,
    train_fold=args.train,
    val_fold=args.val,
    subsampling=args.subsampling,
    subsampling_method=args.subsampling_method,
    verbose=1
)


model_func = get_model_from_name(args.model)
m1 = model_func()
m1.cuda()

# loss and optimizer
criterion_bce = nn.CrossEntropyLoss(reduction="mean")

optimizer = torch.optim.SGD(
    m1.parameters(),
    weight_decay=args.decay,
    lr=args.base_lr
)

# train and val loaders
train_dataset = Dataset(dataset, train=True, val=False, augments=[], cached=True)
val_dataset = Dataset(dataset, train=False, val=True, cached=True)

# training parameters
nb_epoch = 100
batch_size = 32
nb_batch = len(train_dataset) // batch_size

training_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# scheduler
lr_lambda = lambda epoch: 0.5 * (np.cos(np.pi * epoch / nb_epoch) + 1)
lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
callbacks = [lr_scheduler]
# callbacks = []

# tensorboard
title = "%s_%s_Cosd-lr_sgd-%slr-%swd_%de" % (get_datetime(), model_func.__name__, args.base_lr, args.decay, nb_epoch)
tensorboard = SummaryWriter(log_dir="tensorboard/%s/%s" % (args.log_dir, title), comment=model_func.__name__)


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
