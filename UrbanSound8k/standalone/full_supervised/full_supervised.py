
#%%

import os
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NU M_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
import time

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

#%%

from ubs8k.datasetManager import DatasetManager
from ubs8k.datasets import Dataset

import sys
sys.path.append("../..")

from util.utils import reset_seed, get_datetime, get_model_from_name
from util.checkpoint import CheckPoint
from metric_utils.metrics import CategoricalAccuracy, FScore, ContinueAverage

#%% md

# Arguments

#%%

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset_root", default="../../datasets/ubs8k", type=str)
parser.add_argument("--supervised_ratio", default=1.0, type=float)
parser.add_argument("-t", "--train_folds", nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8, 9], type=int)
parser.add_argument("-v", "--val_folds", nargs="+", default=[10], type=int)

parser.add_argument("--model", default="cnn0", type=str)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--nb_epoch", default=50, type=int)
parser.add_argument("--learning_rate", default=0.003, type=int)

parser.add_argument("--checkpoint_path", default="../../model_save/ubs8k/full_supervised", type=str)
parser.add_argument("--resume", action="store_true", default=False)
parser.add_argument("--tensorboard_path", default="../../tensorboard/ubs8k/full_supervised", type=str)
parser.add_argument("--tensorboard_sufix", default="", type=str)

args = parser.parse_args("")

#%% md

# initialisation

#%%

reset_seed(1234)


#%% md

# Prepare the dataset

#%%

audio_root = os.path.join(args.dataset_root, "audio")
metadata_root = os.path.join(args.dataset_root, "metadata")

manager = DatasetManager(
    metadata_root, audio_root,
    folds=(1,2,3,4,5,6,7,8,9,10),
    verbose=2
)

#%%

# prepare the sampler with the specified number of supervised file
train_dataset = Dataset(manager, folds=(1, 2, 3, 4, 5, 6, 7, 8, 9), cached=True)
val_dataset = Dataset(manager, folds=(10, ), cached=True)

#%% md

# Prep model

#%%

torch.cuda.empty_cache()

model_func = get_model_from_name(args.model)
model = model_func()


#%%

from torchsummaryX import summary
input_tensor = torch.zeros((1, 64, 173), dtype=torch.float)

s = summary(model, input_tensor)


#%% md

## Prep training

#%%

# create model
torch.cuda.empty_cache()

model = model_func()
model.cuda()

#%%
s_idx, u_idx = train_dataset.get_su_indexes()
S_sampler = torch.utils.data.SubsetRandomSampler(s_idx)

training_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=S_sampler, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
#%% md
# training parameters

#%%
# losses
loss_ce = nn.CrossEntropyLoss(reduction="mean")

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

# Checkpoint
checkpoint = CheckPoint(model, optimizer, mode="max", name="%s_cnn.torch" % args.checkpoint_path)

# Metrics
fscore_fn = FScore()
acc_fn = CategoricalAccuracy()
avg = ContinueAverage()
reset_metrics = lambda : [m.reset() for m in [fscore_fn, acc_fn, avg]]

# tensorboard
title = "%s_%s_%.1f" % (get_datetime(), model_func.__name__, args.supervised_ratio)
tensorboard = SummaryWriter(log_dir="%s/%s" % (args.tensorboard_path, title), comment=model_func.__name__)

#%% md

## training function

#%%

UNDERLINE_SEQ = "\033[1;4m"
RESET_SEQ = "\033[0m"


header_form = "{:<8.8} {:<6.6} - {:<6.6} - {:<8.8} {:<6.6} - {:<9.9} {:<12.12}| {:<9.9}- {:<6.6}"
value_form  = "{:<8.8} {:<6} - {:<6} - {:<8.8} {:<6.4f} - {:<9.9} {:<10.4f}| {:<9.4f}- {:<6.4f}"

header = header_form.format(
    "", "Epoch", "%", "Losses:", "ce", "metrics: ", "acc", "F1 ","Time"
)


train_form = value_form
val_form = UNDERLINE_SEQ + value_form + RESET_SEQ

print(header)

#%%

def train(epoch):
    start_time = time.time()
    print("")

    reset_metrics()
    model.train()

    for i, (X, y) in enumerate(training_loader):
        X = X.cuda()
        y = y.cuda()

        logits = model(X)
        loss = loss_ce(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.set_grad_enabled(False):
            pred = torch.softmax(logits, dim=1)
            pred_arg = torch.argmax(logits, dim=1)
            y_one_hot = F.one_hot(y, num_classes=10)

            acc = acc_fn(pred_arg, y).mean
            fscore = fscore_fn(pred, y_one_hot).mean
            avg_ce = avg(loss.item()).mean

            # logs
            print(train_form.format(
                "Training: ",
                epoch + 1,
                int(100 * (i + 1) / len(training_loader)),
                "", avg_ce,
                "", acc, fscore,
                time.time() - start_time
            ), end="\r")

    tensorboard.add_scalar("train/Lce", avg_ce, epoch)
    tensorboard.add_scalar("train/f1", fscore, epoch)
    tensorboard.add_scalar("train/acc", acc, epoch)

#%%

def val(epoch):
    start_time = time.time()
    print("")
    reset_metrics()
    model.eval()

    with torch.set_grad_enabled(False):
        for i, (X, y) in enumerate(val_loader):
            X = X.cuda()
            y = y.cuda()

            logits = model(X)
            loss = loss_ce(logits, y)

            pred = torch.softmax(logits, dim=1)
            pred_arg = torch.argmax(logits, dim=1)
            y_one_hot = F.one_hot(y, num_classes=10)

            acc = acc_fn(pred_arg, y).mean
            fscore = fscore_fn(pred, y_one_hot).mean
            avg_ce = avg(loss.item()).mean

            # logs
            print(train_form.format(
                "Validation: ",
                epoch + 1,
                int(100 * (i + 1) / len(val_loader)),
                "", avg_ce,
                "", acc, fscore,
                time.time() - start_time
            ), end="\r")

    tensorboard.add_scalar("val/Lce", avg_ce, epoch)
    tensorboard.add_scalar("val/f1", fscore, epoch)
    tensorboard.add_scalar("val/acc", acc, epoch)

    checkpoint.step(acc)

#%%

print(header)

for e in range(args.nb_epoch):
    train(e)
    val(e)

#%% md

# ♫♪.ılılıll|̲̅̅●̲̅̅|̲̅̅=̲̅̅|̲̅̅●̲̅̅|llılılı.♫♪
