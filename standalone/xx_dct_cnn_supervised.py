import os
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NU M_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
import numpy as np
import time
import math
import random
import argparse
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter


# In[2]:


import sys
sys.path.append("../src/")

from datasetManager import DatasetManager
from generators import CoTrainingDataset
from samplers import CoTrainingSampler

from models import cnn
from metrics import CategoricalAccuracy, Ratio


# # Utils

# ## Arguments

# In[22]:

parser = argparse.ArgumentParser(description='Deep Co-Training for Semi-Supervised Image Recognition')
parser.add_argument("--model", default="cnn", type=str, help="The name of the model to load")
parser.add_argument("-t", "--train_folds", nargs="+", default="1 2 3 4 5 6 7 8 9", required=True, type=int, help="fold to use for training")
parser.add_argument("-v", "--val_folds", nargs="+", default="10", required=True, type=int, help="fold to use for validation")
parser.add_argument("-T", '--tensorboard_dir', required=True, type=str)
parser.add_argument("--nb_view", default=2, type=int, help="Number of supervised view")
parser.add_argument("--ratio", default=0.1, type=float)
parser.add_argument("--subsampling", default=1.0, type=float, help="subsampling ratio")
parser.add_argument("--subsampling_method", default="balance", type=str, help="subsampling method [random|balance]")
parser.add_argument('--batchsize', '-b', default=100, type=int)
parser.add_argument('--lambda_cot_max', default=10, type=int)
parser.add_argument('--lambda_diff_max', default=0.5, type=float)
parser.add_argument('--seed', default=1234, type=int)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--warm_up', default=80.0, type=float)
parser.add_argument('--momentum', default=0.0, type=float)
parser.add_argument('--decay', default=1e-3, type=float)
parser.add_argument('--epsilon', default=0.02, type=float)
parser.add_argument('--num_class', default=10, type=int)
parser.add_argument('--cifar10_dir', default='./data', type=str)
parser.add_argument('--svhn_dir', default='./data', type=str)
parser.add_argument('--checkpoint_dir', default='checkpoint', type=str)
parser.add_argument('--base_lr', default=0.05, type=float)
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--dataset', default='cifar10', type=str, help='choose svhn or cifar10, svhn is not implemented yey')
parser.add_argument("--job_name", default="default", type=str)
args = parser.parse_args()


# Reproducibility
def reset_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark = False
reset_seed(args.seed)


# Utils
def get_datetime():
    now = datetime.datetime.now()
    return str(now)[:10] + "_" + str(now)[11:-7]

# ======== Prepare the data ========
audio_root = "../dataset/audio"
metadata_root = "../dataset/metadata"
manager = DatasetManager(metadata_root, audio_root,
                         train_fold=args.train_folds,
                         val_fold=args.val_folds,
                         subsampling=args.subsampling,
                         subsampling_method=args.subsampling_method,
                         verbose=1)

train_dataset = CoTrainingDataset(manager, args.ratio, train=True, val=False, augments=(), cached=True)
val_dataset = CoTrainingDataset(manager, 1.0, train=False, val=True, cached=True)
train_sampler = CoTrainingSampler(train_dataset, args.batchsize, nb_class=10, nb_view=args.nb_view, ratio=None, method="duplicate") # ratio is manually set here

# ======== Prepare the model =========
model_func = cnn

m1 = model_func()

m1 = m1.cuda()


# ---- Loaders ----
nb_epoch = 100
train_loader = data.DataLoader(train_dataset, batch_sampler=train_sampler)
val_loader = data.DataLoader(val_dataset, batch_size=128)



# -------- optimizers & callbacks & criterion --------
params = m1.parameters()
optimizer = optim.SGD(params, lr=args.base_lr, momentum=args.momentum, weight_decay=args.decay)

lr_lambda = lambda epoch: (1.0 + math.cos((epoch-1)*math.pi/args.epochs))
lr_scheduler = LambdaLR(optimizer, lr_lambda)

criterion = nn.CrossEntropyLoss()
callbacks = [lr_scheduler]


# ---- metrics ----
acc_func = CategoricalAccuracy()

def reset_all_metrics():
    acc_func.reset()

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

# ---- title ----
title = "%s_%s_cnn_Cosd-lr_sgd-%slr-%swd_%de" % (args.job_name, get_datetime(), args.base_lr, args.decay, nb_epoch)
tensorboard = SummaryWriter("tensorboard/%s/%s" % (args.tensorboard_dir, title))


# # Training

# In[93]:


def train(epoch):
    m1.train()
    reset_all_metrics()

    running_loss = 0.0

    start_time = time.time()
    print("")

    for batch, (X, y) in enumerate(train_loader):
        X = [x.squeeze() for x in X]
        y = [y_.squeeze() for y_ in y]

        # separate Supervised (S) and Unsupervised (U) parts
        X_S, X_U = X[:-1], X[-1]
        y_S, y_U = y[:-1], y[-1]

        # Only one view interesting, no U
        X_S = X_S[0]
        y_S = y_S[0]

        X_S, y_S = X_S.cuda().float(), y_S.cuda().long()

        # ======== perform prediction ========
        logits_S = m1(X_S)
        _, pred_S = torch.max(logits_S, 1)

        # ======== calculate loss ========
        loss_sup = criterion(logits_S, y_S)
        total_loss = loss_sup

        # ======== backpropagation =======
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # ======== Calc the metrics ========
        acc = acc_func(pred_S, y_S)
        running_loss += total_loss.item()

        # print statistics
        print("Epoch %s: %.2f%% : train acc: %.3f - Loss: %.3f - time: %.2f" % (
            epoch, (batch / len(train_sampler)) * 100,
            acc,
            running_loss / (batch+1),
            time.time() - start_time,
        ), end="\r")

    # using tensorboard to monitor loss and acc\n",
    tensorboard.add_scalar('train/total_loss', total_loss.item(), epoch)
    tensorboard.add_scalar('train/acc', acc, epoch)

    # Return the total loss to check for NaN
    return total_loss.item()


# In[100]:


def test(epoch):
    m1.eval()

    reset_all_metrics()

    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(val_loader):
            X = X.squeeze()
            y = y.squeeze()

            # separate Supervised (S) and Unsupervised (U) parts
            X = X.cuda()
            y = y.cuda()

            logits = m1(X)
            _, pred = torch.max(logits, 1)

            loss_val = criterion(logits, y)

            acc_val = acc_func(pred, y)

        print("\nEpoch %s: Val acc: %.3f - loss: %.3f" % (
            epoch,
            acc_val,
            loss_val.item()
        ))

    tensorboard.add_scalar("val/acc", acc_val, epoch)
    tensorboard.add_scalar("val/loss", loss_val.item(), epoch)

    tensorboard.add_scalar("detail_hyperparameters/learning_rate", get_lr(optimizer), epoch)

    # Apply callbacks
    for c in callbacks:
        c.step()


# In[101]:


for epoch in range(0, args.epochs):
    total_loss = train(epoch)

    if np.isnan(total_loss):
        print("Losses are NaN, stoping the training here")
        break

    test(epoch)


# # ♫♪.ılılıll|̲̅̅●̲̅̅|̲̅̅=̲̅̅|̲̅̅●̲̅̅|llılılı.♫♪
