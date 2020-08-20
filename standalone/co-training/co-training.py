import os
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NU M_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
import time

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from advertorch.attacks import GradientSignAttack

from ubs8k.datasetManager import DatasetManager
from ubs8k.datasets import Dataset

from DCT.util.utils import reset_seed, get_datetime, get_model_from_name, ZipCycle
from metric_utils.metrics import CategoricalAccuracy, FScore, ContinueAverage, Ratio
from DCT.util.checkpoint import CheckPoint

from DCT.ramps import Warmup, sigmoid_rampup, sigmoid_rampdown
from DCT.losses import loss_cot, loss_diff, loss_sup

# Arguments
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset_root", default="../../datasets/ubs8k", type=str)
parser.add_argument("--supervised_ratio", default=0.1, type=float)
parser.add_argument("--supervised_mult", default=1.0, type=float)
parser.add_argument("-t", "--train_folds", nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8, 9], type=int)
parser.add_argument("-v", "--val_folds", nargs="+", default=[10], type=int)

parser.add_argument("--model", default="cnn03", type=str)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--nb_epoch", default=100, type=int)
parser.add_argument("--learning_rate", default=0.003, type=int)

parser.add_argument("--lambda_sup_max", default=1, type=float)
parser.add_argument("--lambda_cot_max", default=10, type=float)
parser.add_argument("--lambda_diff_max", default=0.5, type=float)
parser.add_argument("--warmup_length", default=80, type=int)
parser.add_argument("--epsilon", default=0.02, type=float)

parser.add_argument("--augment", action="append", help="augmentation. use as if python script")
parser.add_argument("--augment_S", action="store_true", help="Apply augmentation on Supervised part")
parser.add_argument("--augment_U", action="store_true", help="Apply augmentation on Unsupervised part")

parser.add_argument("--checkpoint_path", default="../../model_save/ubs8k/deep-co-training", type=str)
parser.add_argument("--resume", action="store_true", default=False)
parser.add_argument("--tensorboard_path", default="../../tensorboard/ubs8k/deep-co-training", type=str)
parser.add_argument("--tensorboard_sufix", default="", type=str)

args = parser.parse_args()

# %% md

## Prepare the dataset

# %%

audio_root = os.path.join(args.dataset_root, "audio")
metadata_root = os.path.join(args.dataset_root, "metadata")
all_folds = args.train_folds + args.val_folds

manager = DatasetManager(
    metadata_root, audio_root,
    folds=all_folds,
    verbose=2
)

# %%

# prepare the sampler with the specified number of supervised file
train_dataset = Dataset(manager, folds=args.train_folds, cached=True)
val_dataset = Dataset(manager, folds=args.val_folds, cached=True)

# %% md

## Models

# %%

torch.cuda.empty_cache()
model_func = get_model_from_name(args.model)

m1, m2 = model_func(manager=manager), model_func(manager=manager)

m1 = m1.cuda()
m2 = m2.cuda()

# %% md

## Adversarial generator

# %%

# adversarial generation
adv_generator_1 = GradientSignAttack(
    m1, loss_fn=nn.CrossEntropyLoss(reduction="sum"),
    eps=args.epsilon, clip_min=-80, clip_max=0, targeted=True
)

adv_generator_2 = GradientSignAttack(
    m2, loss_fn=nn.CrossEntropyLoss(reduction="sum"),
    eps=args.epsilon, clip_min=-80, clip_max=0, targeted=True
)

# %% md

## Loaders
s_idx, u_idx = train_dataset.split_s_u(args.supervised_ratio)

# Calc the size of the Supervised and Unsupervised batch
nb_s_file = len(s_idx)
nb_u_file = len(u_idx)

ratio = nb_s_file / nb_u_file
s_batch_size = int(np.floor(args.batch_size * ratio))
u_batch_size = int(np.ceil(args.batch_size * (1 - ratio)))


# create the sampler, the loader and "zip" them
sampler_s1 = data.SubsetRandomSampler(s_idx)
sampler_s2 = data.SubsetRandomSampler(s_idx)
sampler_u = data.SubsetRandomSampler(u_idx)

train_loader_s1 = data.DataLoader(train_dataset, batch_size=s_batch_size, sampler=sampler_s1)
train_loader_s2 = data.DataLoader(train_dataset, batch_size=s_batch_size, sampler=sampler_s2)
train_loader_u = data.DataLoader(train_dataset, batch_size=u_batch_size, sampler=sampler_u)

train_loader = ZipCycle([train_loader_s1, train_loader_s2, train_loader_u])
val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

# training parameters
# tensorboard
tensorboard_title = "%s_%s_%.1f" % (get_datetime(), model_func.__name__, args.supervised_ratio)
checkpoint_title = "%s_%.1f" % (model_func.__name__, args.supervised_ratio)
tensorboard = SummaryWriter(log_dir="%s/%s" % (args.tensorboard_path, tensorboard_title), comment=model_func.__name__)

# Losses
# see losses.py

# Optimizer
params = list(m1.parameters()) + list(m2.parameters())
optimizer = torch.optim.Adam(params, lr=args.learning_rate)

# define the warmups
lambda_sup = Warmup(args.lambda_sup_max, args.warmup_length, sigmoid_rampdown)
lambda_cot = Warmup(args.lambda_cot_max, args.warmup_length, sigmoid_rampup)
lambda_diff = Warmup(args.lambda_diff_max, args.warmup_length, sigmoid_rampup)

# callback
lr_lambda = lambda epoch: (1.0 + np.cos((epoch - 1) * np.pi / args.nb_epoch))
lr_scheduler = LambdaLR(optimizer, lr_lambda)
callbacks = [lr_scheduler, lambda_sup, lambda_cot, lambda_diff]

# checkpoints
checkpoint_m1 = CheckPoint(m1, optimizer, mode="max", name="%s/%s_m1.torch" % (args.checkpoint_path, checkpoint_title))
checkpoint_m2 = CheckPoint(m2, optimizer, mode="max", name="%s/%s_m2.torch" % (args.checkpoint_path, checkpoint_title))

# metrics
metrics_fn = dict(
    ratio_s=[Ratio(), Ratio()],
    ratio_u=[Ratio(), Ratio()],
    acc_s=[CategoricalAccuracy(), CategoricalAccuracy()],
    acc_u=[CategoricalAccuracy(), CategoricalAccuracy()],
    f1_s=[FScore(), FScore()],
    f1_u=[FScore(), FScore()],

    avg_total=ContinueAverage(),
    avg_sup=ContinueAverage(),
    avg_cot=ContinueAverage(),
    avg_diff=ContinueAverage(),
)


def reset_metrics():
    for item in metrics_fn.values():
        if isinstance(item, list):
            for f in item:
                f.reset()
        else:
            item.reset()
reset_metrics()

## Can resume previous training
if args.resume:
    checkpoint_m1.load_last()


## Metrics and hyperparameters
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


# %% md

# Training functions

# %%

UNDERLINE_SEQ = "\033[1;4m"

RESET_SEQ = "\033[0m"

header_form = "{:<8.8} {:<6.6} - {:<6.6} - {:<8.8} {:<6.6} | {:<6.6} | {:<6.6} | {:<6.6} - {:<9.9} {:<9.9} | {:<9.9}- {:<6.6}"
value_form = "{:<8.8} {:<6} - {:<6} - {:<8.8} {:<6.4f} | {:<6.4f} | {:<6.4f} | {:<6.4f} - {:<9.9} {:<9.4f} | {:<9.4f}- {:<6.4f}"

header = header_form.format(
    "", "Epoch", "%", "Losses:", "Lsup", "Lcot", "Ldiff", "total", "metrics: ", "acc_s1", "acc_u1", "Time"
)

train_form = value_form
val_form = UNDERLINE_SEQ + value_form + RESET_SEQ

print(header)

def maximum():
    def func(key, value):
        if key not in func.max:
            func.max[key] = value
        else:
            if func.max[key] < value:
                func.max[key] = value
        return func.max[key]

    func.max = dict()
    return func
maximum_fn = maximum()

def train(epoch):
    start_time = time.time()
    print("")

    reset_metrics()
    m1.train()
    m2.train()

    for batch, (S1, S2, U) in enumerate(train_loader):
        x_s1, y_s1 = S1
        x_s2, y_s2 = S2
        x_u, y_u = U

        x_s1, x_s2, x_u = x_s1.cuda(), x_s2.cuda(), x_u.cuda()
        y_s1, y_s2, y_u = y_s1.cuda(), y_s2.cuda(), y_u.cuda()

        logits_s1 = m1(x_s1)
        logits_s2 = m2(x_s2)
        logits_u1 = m1(x_u)
        logits_u2 = m2(x_u)

        # pseudo labels of U
        pred_u1 = torch.argmax(logits_u1, 1)
        pred_u2 = torch.argmax(logits_u2, 1)

        # ======== Generate adversarial examples ========
        # fix batchnorm ----
        m1.eval()
        m2.eval()

        # generate adversarial examples ----
        adv_data_s1 = adv_generator_1.perturb(x_s1, y_s1)
        adv_data_u1 = adv_generator_1.perturb(x_u, pred_u1)

        adv_data_s2 = adv_generator_2.perturb(x_s2, y_s2)
        adv_data_u2 = adv_generator_2.perturb(x_u, pred_u2)

        m1.train()
        m2.train()

        # predict adversarial examples ----
        adv_logits_s1 = m1(adv_data_s2)
        adv_logits_s2 = m2(adv_data_s1)

        adv_logits_u1 = m1(adv_data_u2)
        adv_logits_u2 = m2(adv_data_u1)

        # ======== calculate the differents loss ========
        # zero the parameter gradients ----
        optimizer.zero_grad()
        m1.zero_grad()
        m2.zero_grad()

        # losses ----
        l_sup = loss_sup(logits_s1, logits_s2, y_s1, y_s2)

        l_cot = loss_cot(logits_u1, logits_u2)

        l_diff = loss_diff(
            logits_s1, logits_s2, adv_logits_s1, adv_logits_s2,
            logits_u1, logits_u2, adv_logits_u1, adv_logits_u2
        )

        total_loss = lambda_sup() * l_sup + lambda_cot() * l_cot + lambda_diff() * l_diff
        total_loss.backward()
        optimizer.step()

        # ======== Calc the metrics ========
        with torch.set_grad_enabled(False):
            # accuracies ----
            pred_s1 = torch.argmax(logits_s1, dim=1)
            pred_s2 = torch.argmax(logits_s2, dim=1)

            acc_s1 = metrics_fn["acc_s"][0](pred_s1, y_s1)
            acc_s2 = metrics_fn["acc_s"][1](pred_s2, y_s2)
            acc_u1 = metrics_fn["acc_u"][0](pred_u1, y_u)
            acc_u2 = metrics_fn["acc_u"][1](pred_u2, y_u)

            # ratios  ----
            adv_pred_s1 = torch.argmax(adv_logits_s1, 1)
            adv_pred_s2 = torch.argmax(adv_logits_s2, 1)
            adv_pred_u1 = torch.argmax(adv_logits_u1, 1)
            adv_pred_u2 = torch.argmax(adv_logits_u2, 1)

            ratio_s1 = metrics_fn["ratio_s"][0](adv_pred_s1, y_s1)
            ratio_s2 = metrics_fn["ratio_s"][0](adv_pred_s2, y_s2)
            ratio_u1 = metrics_fn["ratio_s"][0](adv_pred_u1, y_u)
            ratio_u2 = metrics_fn["ratio_s"][0](adv_pred_u2, y_u)
            # ========

            avg_total = metrics_fn["avg_total"](total_loss.item())
            avg_sup = metrics_fn["avg_sup"](l_sup.item())
            avg_diff = metrics_fn["avg_diff"](l_diff.item())
            avg_cot = metrics_fn["avg_cot"](l_cot.item())

            # logs
            print(train_form.format(
                "Training: ",
                epoch + 1,
                int(100 * (batch + 1) / len(train_loader)),
                "", avg_sup.mean, avg_cot.mean, avg_diff.mean, avg_total.mean,
                "", acc_s1.mean, acc_u1.mean,
                time.time() - start_time
            ), end="\r")

    # using tensorboard to monitor loss and acc\n",
    tensorboard.add_scalar('train/total_loss', avg_total.mean, epoch)
    tensorboard.add_scalar('train/Lsup', avg_sup.mean, epoch)
    tensorboard.add_scalar('train/Lcot', avg_cot.mean, epoch)
    tensorboard.add_scalar('train/Ldiff', avg_diff.mean, epoch)
    tensorboard.add_scalar("train/acc_1", acc_s1.mean, epoch)
    tensorboard.add_scalar("train/acc_2", acc_s2.mean, epoch)

    tensorboard.add_scalar("detail_acc/acc_s1", acc_s1.mean, epoch)
    tensorboard.add_scalar("detail_acc/acc_s2", acc_s2.mean, epoch)
    tensorboard.add_scalar("detail_acc/acc_u1", acc_u1.mean, epoch)
    tensorboard.add_scalar("detail_acc/acc_u2", acc_u2.mean, epoch)

    tensorboard.add_scalar("detail_ratio/ratio_s1", ratio_s1.mean, epoch)
    tensorboard.add_scalar("detail_ratio/ratio_s2", ratio_s2.mean, epoch)
    tensorboard.add_scalar("detail_ratio/ratio_u1", ratio_u1.mean, epoch)
    tensorboard.add_scalar("detail_ratio/ratio_u2", ratio_u2.mean, epoch)

    # Return the total loss to check for NaN
    return total_loss.item(), msg


# %%

def test(epoch, msg=""):
    start_time = time.time()
    print("")

    reset_metrics()
    m1.eval()
    m2.eval()

    with torch.set_grad_enabled(False):
        for batch, (X, y) in enumerate(val_loader):
            x = X.cuda()
            y = y.cuda()

            logits_1 = m1(x)
            logits_2 = m2(x)

            # losses ----
            l_sup = loss_sup(logits_1, logits_2, y, y)

            # ======== Calc the metrics ========
            # accuracies ----
            pred_1 = torch.argmax(logits_1, dim=1)
            pred_2 = torch.argmax(logits_2, dim=1)

            acc_1 = metrics_fn["acc_s"][0](pred_1, y)
            acc_2 = metrics_fn["acc_s"][1](pred_2, y)

            avg_sup = metrics_fn["avg_sup"](l_sup.item())

            # logs
            print(val_form.format(
                "Validation: ",
                epoch + 1,
                int(100 * (batch + 1) / len(train_loader)),
                "", avg_sup.mean, 0.0, 0.0, avg_sup.mean,
                "", acc_1.mean, 0.0,
                time.time() - start_time
            ), end="\r")

    tensorboard.add_scalar("val/acc_1", acc_1.mean, epoch)
    tensorboard.add_scalar("val/acc_2", acc_2.mean, epoch)

    tensorboard.add_scalar("max/acc_1", maximum_fn("acc_1", acc_1.mean), epoch)
    tensorboard.add_scalar("max/acc_2", maximum_fn("acc_2", acc_2.mean), epoch)

    tensorboard.add_scalar("detail_hyperparameters/lambda_cot", lambda_cot(), epoch)
    tensorboard.add_scalar("detail_hyperparameters/lambda_diff", lambda_diff(), epoch)
    tensorboard.add_scalar("detail_hyperparameters/learning_rate", get_lr(optimizer), epoch)

    # Apply callbacks
    for c in callbacks:
        c.step()

    # call checkpoint
    checkpoint_m1.step(acc_1.mean)
    checkpoint_m2.step(acc_2.mean)


# %%

print(header)

for epoch in range(0, args.nb_epoch):
    total_loss, msg = train(epoch)

    if np.isnan(total_loss):
        print("Losses are NaN, stoping the training here")
        break

    test(epoch, msg)

tensorboard.flush()
tensorboard.close()

# %%


