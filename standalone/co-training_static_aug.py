import os
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NU M_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
import numpy as np
import time
import math
import argparse
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
from advertorch.attacks import GradientSignAttack

from ubs8k.datasetManager import StaticManager # <-- static manager allow usage of static augmentation store in a specific hdf file
from ubs8k.generators import CoTrainingDataset
from ubs8k.samplers import CoTrainingSampler
from ubs8k.utils import get_datetime, get_model_from_name, reset_seed, set_logs

from ubs8k.losses import loss_cot, p_loss_diff, p_loss_sup
from ubs8k.metrics import CategoricalAccuracy, Ratio
from ubs8k.ramps import Warmup, sigmoid_rampup

import ubs8k.img_augmentations
import ubs8k.spec_augmentations
import ubs8k.signal_augmentations

# ---- Arguments ----
parser = argparse.ArgumentParser(description='Deep Co-Training for Semi-Supervised Image Recognition')
parser.add_argument("--model", default="cnn", type=str, help="Model to load, see list of model in models.py")
parser.add_argument("-t", "--train_folds", nargs="+", default="1 2 3 4 5 6 7 8 9", required=True, type=int, help="fold to use for training")
parser.add_argument("-v", "--val_folds", nargs="+", default="10", type=int, required=True, help="fold to use for validation")
parser.add_argument("--nb_view", default=2, type=int, help="Number of supervised view")
parser.add_argument("--ratio", default=0.1, type=float)
parser.add_argument("--parser_ratio", default=None, type=float, help="ratio to apply for sampling the S and U data")
parser.add_argument("--subsampling", default=1.0, type=float, help="subsampling ratio")
parser.add_argument("--subsampling_method", default="balance", type=str, help="method to perform subsampling [random | balance]")
parser.add_argument('--batchsize', '-b', default=100, type=int)
parser.add_argument('--lambda_cot_max', default=10, type=int)
parser.add_argument('--lambda_diff_max', default=0.5, type=float)
parser.add_argument('--seed', default=1234, type=int)
parser.add_argument('--epochs', default=600, type=int)
parser.add_argument('--warm_up', default=80.0, type=float)
parser.add_argument('--momentum', default=0.0, type=float)
parser.add_argument('--decay', default=1e-3, type=float)
parser.add_argument('--epsilon', default=0.02, type=float)
parser.add_argument('--num_class', default=10, type=int)
parser.add_argument("-T", '--tensorboard_dir', default='tensorboard/', type=str)
parser.add_argument('--checkpoint_dir', default='checkpoint', type=str)
parser.add_argument('--base_lr', default=0.05, type=float)
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--dataset', default='cifar10', type=str, help='choose svhn or cifar10, svhn is not implemented yey')
parser.add_argument("--job_name", default="default", type=str)
parser.add_argument("--audio_root", default="../dataset/audio")
parser.add_argument("--metadata_root", default="../dataset/metadata")
parser.add_argument("--augmentation_file", default="../dataset/audio/")
parser.add_argument("-a","--augments", action="append", help="Augmentation. use as if python script. Must be a str")
parser.add_argument("-sa", "--static_augments", default="{}", type=str, help="a valid dictionnary where key are the augmentation to use and values their ratio")
parser.add_argument("--augment_S", action="store_true", help="Apply augmentation on Supervised part")
parser.add_argument("--augment_U", action="store_true", help="Apply augmentation on Unsupervised part")
parser.add_argument("--num_workers", default=0, type=int, help="Choose number of worker to train the model")
parser.add_argument("--log", default="warning", help="Log level")
args = parser.parse_args()

# Logging system
set_logs(args.log)

# Reproducibility
reset_seed(args.seed)

# ---- Prepare augmentation ----
# list of dynamic augmentation
dynamic_augments = [] if args.augments is None else list(map(eval, args.augments))

# list of static augmentation
static_augments = eval(args.static_augments)


# ======== Prepare the data ========
audio_root = "../dataset/audio"
metadata_root = "../dataset/metadata"
augmentation_file = os.path.join(audio_root, "urbansound8k_22050_augmentation.hdf5")

manager = StaticManager(
    metadata_root, audio_root,
    static_augment_file=augmentation_file, static_augment_list=list(static_augments.keys()),
    subsampling=args.subsampling, subsampling_method=args.subsampling_method,
    train_fold=args.train_folds, val_fold=args.val_folds,
    verbose=1
)

# prepare the sampler with the specified number of supervised file
train_dataset = CoTrainingDataset(manager, args.ratio, train=True, val=False, augments=dynamic_augments, static_augmentation=static_augments, S_augment=args.augment_S, U_augment=args.augment_U, cached=True)
val_dataset = CoTrainingDataset(manager, 1.0, train=False, val=True, cached=True)
sampler = CoTrainingSampler(train_dataset, args.batchsize, nb_class=10, nb_view=args.nb_view, ratio=args.parser_ratio, method="duplicate") # ratio is automatically set here.

# ======== Prepare the model ========
model_func = get_model_from_name(args.model)
m1 = model_func(dataset=manager)
m2 = model_func(dataset=manager)

m1 = m1.cuda()
m2 = m2.cuda()

# ======== Loaders & adversarial generators ========
train_loader = data.DataLoader(train_dataset, batch_sampler=sampler, num_workers=args.num_workers)
val_loader = data.DataLoader(val_dataset, batch_size=128)

# adversarial generation
input_max_value = 0
input_min_value = -80
adv_generator_1 = GradientSignAttack(
    m1, loss_fn=nn.CrossEntropyLoss(reduction="sum"),
    eps=args.epsilon, clip_min=input_min_value, clip_max=input_max_value, targeted=False
)

adv_generator_2 = GradientSignAttack(
    m2, loss_fn=nn.CrossEntropyLoss(reduction="sum"),
    eps=args.epsilon, clip_min=input_min_value, clip_max=input_max_value, targeted=False
)


# ======== optimizers & callbacks =========
params = list(m1.parameters()) + list(m2.parameters())
optimizer = optim.SGD(params, lr=args.base_lr, momentum=args.momentum, weight_decay=args.decay)

lr_lambda = lambda epoch: (1.0 + math.cos((epoch-1)*math.pi/args.epochs))
lr_scheduler = LambdaLR(optimizer, lr_lambda)

callbacks = [lr_scheduler]


# ========= Metrics and hyperparameters ========
# define the metrics
ratioS = [Ratio(), Ratio()]
ratioU = [Ratio(), Ratio()]
ratioSU = [Ratio(), Ratio()]
accS = [CategoricalAccuracy(), CategoricalAccuracy()]
accU = [CategoricalAccuracy(), CategoricalAccuracy()]
accSU = [CategoricalAccuracy(), CategoricalAccuracy()]

# define the warmups
lambda_cot = Warmup(args.lambda_cot_max, args.warm_up, sigmoid_rampup)
lambda_diff = Warmup(args.lambda_diff_max, args.warm_up, sigmoid_rampup)

def reset_all_metrics():
    all_metrics = [*ratioS, *ratioU, *ratioSU, *accS, *accU, *accSU]
    for m in all_metrics:
        m.reset()

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

title = "%s_%s_%s_%sss_%slr_%se_%slcm_%sldm_%swl_%swd" % (
    get_datetime(),
    args.job_name,
    model_func.__name__,
    args.subsampling,
    args.base_lr,
    args.epsilon,
    args.lambda_cot_max,
    args.lambda_diff_max,
    args.warm_up,
    args.decay
)
tensorboard = SummaryWriter("tensorboard/%s/%s" % (args.tensorboard_dir, title))


# ======== Training ========
def train(epoch):
    m1.train()
    m2.train()
    reset_all_metrics()

    running_loss = 0.0
    ls = 0.0
    lc = 0.0
    ld = 0.0

    reset_all_metrics()

    start_time = time.time()
    print("")

    for batch, (X, y) in enumerate(train_loader):
        X = [x.squeeze() for x in X]
        y = [y_.squeeze() for y_ in y]

        # separate Supervised (S) and Unsupervised (U) parts
        X_S, X_U = X[:-1], X[-1]
        y_S, y_U = y[:-1], y[-1]

        for i in range(len(X_S)):
            X_S[i] = X_S[i].cuda()
            y_S[i] = y_S[i].cuda()
        X_U, y_U = X_U.cuda(), y_U.cuda()

        logits_S1 = m1(X_S[0])
        logits_S2 = m2(X_S[1])
        logits_U1 = m1(X_U)
        logits_U2 = m2(X_U)

        _, pred_S1 = torch.max(logits_S1, 1)
        _, pred_S2 = torch.max(logits_S2, 1)

        # pseudo labels of U
        _, pred_U1 = torch.max(logits_U1, 1)
        _, pred_U2 = torch.max(logits_U2, 1)

        # ======== Generate adversarial examples ========
        # fix batchnorm ----
        m1.eval()
        m2.eval()

        #generate adversarial examples ----
        adv_data_S1 = adv_generator_1.perturb(X_S[0], y_S[0])
        adv_data_U1 = adv_generator_1.perturb(X_U, pred_U1)

        adv_data_S2 = adv_generator_2.perturb(X_S[1], y_S[1])
        adv_data_U2 = adv_generator_2.perturb(X_U, pred_U2)

        m1.train()
        m2.train()

        # predict adversarial examples ----
        adv_logits_S1 = m1(adv_data_S2)
        adv_logits_S2 = m2(adv_data_S1)

        adv_logits_U1 = m1(adv_data_U2)
        adv_logits_U2 = m2(adv_data_U1)

        # ======== calculate the differents loss ========
        # zero the parameter gradients ----
        optimizer.zero_grad()
        m1.zero_grad()
        m2.zero_grad()

        # losses ----
        Loss_sup_S1, Loss_sup_S2, Loss_sup = p_loss_sup(logits_S1, logits_S2, y_S[0], y_S[1])
        Loss_cot = loss_cot(logits_U1, logits_U2)
        pld_S, pld_U, Loss_diff = p_loss_diff(logits_S1, logits_S2, adv_logits_S1, adv_logits_S2, logits_U1, logits_U2, adv_logits_U1, adv_logits_U2)

        total_loss = Loss_sup + lambda_cot() * Loss_cot + lambda_diff() * Loss_diff
        total_loss.backward()
        optimizer.step()

        # ======== Calc the metrics ========
        # accuracies ----
        pred_SU1 = torch.cat((pred_S1, pred_U1), 0)
        pred_SU2 = torch.cat((pred_S2, pred_U2), 0)
        y_SU1 = torch.cat((y_S[0], y_U), 0)
        y_SU2 = torch.cat((y_S[1], y_U), 0)

        acc_S1 = accS[0](pred_S1, y_S[0])
        acc_S2 = accS[1](pred_S2, y_S[1])
        acc_U1 = accU[0](pred_U1, y_U)
        acc_U2 = accU[1](pred_U2, y_U)
        acc_SU1 = accSU[0](pred_SU1, y_SU1)
        acc_SU2 = accSU[1](pred_SU2, y_SU2)

        # ratios  ----
        _, adv_pred_S1 = torch.max(adv_logits_S1, 1)
        _, adv_pred_S2 = torch.max(adv_logits_S2, 1)
        _, adv_pred_U1 = torch.max(adv_logits_U1, 1)
        _, adv_pred_U2 = torch.max(adv_logits_U2, 1)

        adv_pred_SU1 = torch.cat((adv_pred_S1, adv_pred_U1), 0)
        adv_pred_SU2 = torch.cat((adv_pred_S2, adv_pred_U2), 0)
        adv_y_SU1 = torch.cat((y_S[0], pred_U1), 0)
        adv_y_SU2 = torch.cat((y_S[1], pred_U2), 0)

        ratio_S1 = ratioS[0](adv_pred_S1, y_S[0])
        ratio_S2 = ratioS[1](adv_pred_S2, y_S[1])
        ratio_U1 = ratioU[0](adv_pred_U1, pred_U1)
        ratio_U2 = ratioU[1](adv_pred_U2, pred_U2)
        ratio_SU1 = ratioSU[0](adv_pred_SU1, adv_y_SU1)
        ratio_SU2 = ratioSU[1](adv_pred_SU2, adv_y_SU2)
        # ========

        running_loss += total_loss.item()
        ls += Loss_sup.item()
        lc += Loss_cot.item()
        ld += Loss_diff.item()

        # print statistics
        print("Epoch %s: %.2f%% : train acc: %.3f %.3f - Loss: %.3f %.3f %.3f %.3f - time: %.2f" % (
            epoch, (batch / len(sampler)) * 100,
            acc_SU1, acc_SU2,
            running_loss/(batch+1), ls/(batch+1), lc/(batch+1), ld/(batch+1),
            time.time() - start_time,
        ), end="\r")

    # using tensorboard to monitor loss and acc\n",
    tensorboard.add_scalar('train/total_loss', total_loss.item(), epoch)
    tensorboard.add_scalar('train/Lsup', Loss_sup.item(), epoch )
    tensorboard.add_scalar('train/Lcot', Loss_cot.item(), epoch )
    tensorboard.add_scalar('train/Ldiff', Loss_diff.item(), epoch )
    tensorboard.add_scalar("train/acc_SU1", acc_SU1, epoch )
    tensorboard.add_scalar("train/acc_SU2", acc_SU2, epoch )

    tensorboard.add_scalar("detail_loss/Lsup_S1", Loss_sup_S1.item(), epoch)
    tensorboard.add_scalar("detail_loss/Lsup_S2", Loss_sup_S2.item(), epoch)
    tensorboard.add_scalar("detail_loss/Ldiff_S", pld_S.item(), epoch)
    tensorboard.add_scalar("detail_loss/Ldiff_U", pld_U.item(), epoch)

    tensorboard.add_scalar("detail_acc/acc_S1", acc_S1, epoch)
    tensorboard.add_scalar("detail_acc/acc_S2", acc_S2, epoch)
    tensorboard.add_scalar("detail_acc/acc_U1", acc_U1, epoch)
    tensorboard.add_scalar("detail_acc/acc_U2", acc_U2, epoch)

    tensorboard.add_scalar("detail_ratio/ratio_S1", ratio_S1, epoch)
    tensorboard.add_scalar("detail_ratio/ratio_S2", ratio_S2, epoch)
    tensorboard.add_scalar("detail_ratio/ratio_U1", ratio_U1, epoch)
    tensorboard.add_scalar("detail_ratio/ratio_U2", ratio_U2, epoch)
    tensorboard.add_scalar("detail_ratio/ratio_SU1", ratio_SU1, epoch)
    tensorboard.add_scalar("detail_ratio/ratio_SU2", ratio_SU2, epoch)

    # Return the total loss to check for NaN
    return total_loss.item()


def test(epoch):
    global best_acc
    m1.eval()
    m2.eval()
    reset_all_metrics()

    correct1 = 0
    correct2 = 0
    total1 = 0
    total2 = 0

    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(val_loader):
            X = X.squeeze()
            y = y.squeeze()

            # separate Supervised (S) and Unsupervised (U) parts
            X = X.cuda()
            y = y.cuda()

            outputs1 = m1(X)
            predicted1 = outputs1.max(1)
            total1 += y.size(0)
            correct1 += predicted1[1].eq(y).sum().item()

            outputs2 = m2(X)
            predicted2 = outputs2.max(1)
            total2 += y.size(0)
            correct2 += predicted2[1].eq(y).sum().item()

    print('\nnet1 test acc: %.3f%% (%d/%d) | net2 test acc: %.3f%% (%d/%d)'
        % (100.*correct1/total1, correct1, total1, 100.*correct2/total2, correct2, total2))

    tensorboard.add_scalar("val/acc_1", correct1 / total1, epoch)
    tensorboard.add_scalar("val/acc_2", correct2 / total2, epoch)

    tensorboard.add_scalar("detail_hyperparameters/lambda_cot", lambda_cot(), epoch)
    tensorboard.add_scalar("detail_hyperparameters/lambda_diff", lambda_diff(), epoch)
    tensorboard.add_scalar("detail_hyperparameters/learning_rate", get_lr(optimizer), epoch)

    # Apply callbacks and warmup
    for c in callbacks:
        c.step()
    lambda_cot.step()
    lambda_diff.step()


for epoch in range(0, args.epochs):
    total_loss = train(epoch)
    if np.isnan(total_loss):
        print("Losses are NaN, stoping the training here")
        break
    test(epoch)

# # ♫♪.ılılıll|̲̅̅●̲̅̅|̲̅̅=̲̅̅|̲̅̅●̲̅̅|llılılı.♫♪
