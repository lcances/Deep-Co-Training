import os
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NU M_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
import numpy as np
import tqdm
import time
import random
import sys
import argparse
sys.path.append("../../src/")

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

from datasetManager import DatasetManager
from generators import Generator
from metrics import CategoricalAccuracy
from models import ScalableCnn1


# ======================================================================================================================
#           PARAMETERS
# ======================================================================================================================
parser = argparse.ArgumentParser(description='Deep Co-Training for Semi-Supervised Image Recognition')
parser.add_argument("-t", "--train_folds", nargs="*", default=[1, 2, 3, 4, 5, 6, 7, 8, 9], type=int, help="fold to use for training")
parser.add_argument("-v", "--val_folds", nargs="*", default=[10], type=int, help="fold to use for validation")
parser.add_argument('--batchsize', default=100, type=int)
parser.add_argument('--seed', default=1234, type=int)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--momentum', default=0.0, type=float)
parser.add_argument('--decay', default=1e-2, type=float)
parser.add_argument('--num_class', default=10, type=int)
parser.add_argument('--tensorboard_dir', default='tensorboard/compoundScaling', type=str)
parser.add_argument('--checkpoint_dir', default='checkpoint', type=str)
parser.add_argument('--base_lr', default=0.05, type=float)
parser.add_argument("--job_name", default="default", type=str)

parser.add_argument("-a", "--alpha", default=1, type=float)
parser.add_argument("-b", "--beta", default=1, type=float)
parser.add_argument("-g", "--gamma", default=1, type=float)
parser.add_argument("--init_conv_inputs", nargs="*", default=[1, 32, 64, 64], type=int)
parser.add_argument("--init_conv_outputs", nargs="*", default=[32, 64, 64, 64], type=int)
parser.add_argument("--init_linear_inputs", nargs="*", default=[1344], type=int)
parser.add_argument("--init_linear_outputs", nargs="*", default=[10], type=int)
parser.add_argument("--init_resolution", nargs="*", default=[64, 173], type=int)
args = parser.parse_args()

# ======================================================================================================================
#           UTILITIES
# ======================================================================================================================
import datetime
def get_datetime():
    now = datetime.datetime.now()
    return str(now)[:10] + "_" + str(now)[11:-7]

def reset_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark = False

reset_seed(args.seed)

def get_valid_scaling_factors():
    alpha = np.linspace(1, 2, 6)
    beta = np.linspace(1, 2, 6)
    gamma = np.linspace(1, 1, 1)

    import itertools

    valid_scaling_factors = []
    for a, b, g in itertools.product(alpha, beta, gamma):
        M = a * b**2 * g**2

        if M <= 2:
            valid_scaling_factors.append((a, b, g))

    return valid_scaling_factors

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
# cudnn.benchmark = True



# ======================================================================================================================
#           PREPARATION TRAINING
# ======================================================================================================================
torch.cuda.empty_cache()

# prep dataset --------
audio_root = "../../dataset/audio"
metadata_root = "../../dataset/metadata"

dataset = DatasetManager(
    metadata_root, audio_root,
    train_fold=args.train_folds, val_fold=args.val_folds,
    verbose=1
)

# ScallableCNN --------
model_func = ScalableCnn1
m1 = model_func(
    dataset,
    compound_scales=(args.alpha, args.beta, args.gamma),
    initial_conv_inputs=args.init_conv_inputs,
    initial_conv_outputs=args.init_conv_outputs,
    initial_linear_inputs=args.init_linear_inputs,
    initial_linear_outputs=args.init_linear_outputs,
    initial_resolution=args.init_resolution,
)
m1 = m1.cuda()

# loss and optimizer --------
criterion_bce = nn.CrossEntropyLoss(reduction="mean")

optimizer = torch.optim.SGD(
    m1.parameters(),
    weight_decay=args.decay,
    lr=args.base_lr
)

# Augmentation to use --------
augments = []

# train and val loaders --------
train_dataset = Generator(dataset, augments=augments)

x, y = train_dataset.validation
x = torch.from_numpy(x)
y = torch.from_numpy(y)
val_dataset = torch.utils.data.TensorDataset(x, y)

# training parameters
nb_epoch = args.epochs
batch_size = args.batchsize
nb_batch = len(train_dataset) // batch_size

training_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# scheduler
lr_lambda = lambda epoch: 0.05 * (np.cos(np.pi * epoch / nb_epoch) + 1)
lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
callbacks = [lr_scheduler]
callbacks = []

# tensorboard
title = "%s_%s_%s_Cosd-%slr_%se_%sa_%sb_%sg_noaugment" % (
    get_datetime(),
    model_func.__name__,
    args.job_name,
    args.base_lr,
    nb_epoch,
    args.alpha, args.beta, args.gamma,
)
tensorboard = SummaryWriter(log_dir="%s/%s" % (args.tensorboard_dir, title), comment=model_func.__name__)



# ======================================================================================================================
#           TRAINING
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

        weak_loss = criterion_bce(logits, y)

        total_loss = weak_loss

        # calc metrics
#         y_pred = torch.log_softmax(logits, dim=1)
        _, y_pred = torch.max(logits, 1)
        acc = acc_func(y_pred, y)

        # ======== back propagation ========
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # ======== history ========
        print("Epoch {}, {:d}% \t ce: {:.4f} - acc: {:.4f} - took: {:.2f}s".format(
            epoch+1,
            int(100 * (i+1) / nb_batch),
            total_loss.item(),
            acc,
            time.time() - start_time
        ),end="\r")

    # using tensorboard to monitor loss and acc
    tensorboard.add_scalar('train/ce', total_loss.item(), epoch)
    tensorboard.add_scalar("train/acc", 100. * acc, epoch )

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

            #Print statistics
            print("Epoch {}, {:d}% \t ce: {:.4f} - acc: {:.4f} - ce val: {:.4f} - acc val: {:.4f} - took: {:.2f}s".format(
                epoch+1,
                int(100 * (i+1) / nb_batch),
                total_loss.item(),
                acc,
                weak_loss_val.item(),
                acc_val,
                time.time() - start_time
            ),end="\r")

        # using tensorboard to monitor loss and acc
        tensorboard.add_scalar('validation/ce', weak_loss_val.item(), epoch)
        tensorboard.add_scalar("validation/acc", 100. * acc_val, epoch )

    for callback in callbacks:
        callback.step()

# ♫♪.ılılıll|̲̅̅●̲̅̅|̲̅̅=̲̅̅|̲̅̅●̲̅̅|llılılı.♫♪

