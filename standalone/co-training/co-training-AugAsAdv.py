import os
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NU M_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
import time

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from advertorch.attacks import GradientSignAttack

from ubs8k.datasetManager import DatasetManager
from ubs8k.datasets import Dataset

from DCT.util.utils import reset_seed, get_datetime, get_model_from_name, ZipCycle
from metric_utils.metrics import CategoricalAccuracy, FScore, ContinueAverage, Ratio
from DCT.util.checkpoint import CheckPoint

from DCT.ramps import Warmup, sigmoid_rampup
from DCT.losses import loss_cot, loss_diff, loss_sup

import augmentation_utils.spec_augmentations as spec_aug
from DCT.augmentation_list import augmentations


# # Arguments



import argparse
parser = argparse.ArgumentParser()


parser.add_argument("-d", "--dataset_root", default="../datasets", type=str)
parser.add_argument("-D", "--dataset", default="ubs8k", type=str, help="available [ubs8k | cifar10]")
# parser.add_argument("--supervised_mult", default=1.0, type=float)

group_t = parser.add_argument_group("Commun parameters")
group_t.add_argument("-m", "--model", default="cnn03", type=str)
group_t.add_argument("--supervised_ratio", default=0.1, type=float)
group_t.add_argument("--batch_size", default=100, type=int)
group_t.add_argument("--nb_epoch", default=300, type=int)
group_t.add_argument("--learning_rate", default=0.003, type=float)
group_t.add_argument("--resume", action="store_true", default=False)
group_t.add_argument("--seed", default=1234, type=int)

group_u = parser.add_argument_group("UrbanSound8k parameters")
group_u.add_argument("-t", "--train_folds", nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8, 9], type=int)
group_u.add_argument("-v", "--val_folds", nargs="+", default=[10], type=int)

group_c = parser.add_argument_group("Cifar10 parameters")

group_h = parser.add_argument_group('hyperparameters')
group_h.add_argument("--lambda_cot_max", default=10, type=float)
group_h.add_argument("--lambda_diff_max", default=0.5, type=float)
group_h.add_argument("--warmup_length", default=80, type=int)

group_a = parser.add_argument_group("Augmentation")
group_a.add_argument("--augment_m1", default="n_10", help="augmentation. use as if python script")
group_a.add_argument("--augment_m2", default="usn_12", help="augmentation. use as if python script")

group_l = parser.add_argument_group("Logs")
group_l.add_argument("--checkpoint_path", default="../model_save/ubs8k/deep-co-training_aug4adv/test", type=str)
group_l.add_argument("--tensorboard_path", default="../tensorboard/ubs8k/deep-co-training_aug4adv/test", type=str)
group_l.add_argument("--tensorboard_sufix", default="", type=str)

args = parser.parse_args("")


# In[42]:


# modify checkpoint and tensorboard path to fit the dataset
checkpoint_path_ = args.checkpoint_path.split("/")
tensorboard_path_ = args.tensorboard_path.split("/")

checkpoint_path_[2] = args.dataset
tensorboard_path_[2] = args.dataset

args.checkpoint_path = "/".join(checkpoint_path_)
args.tensorboard_path = "/".join(tensorboard_path_)
args


# In[43]:


augmentation_list = list(augmentations.keys())


# In[44]:


reset_seed(1234)


# # Prepare the dataset and the dataloader
# Train_laoder will return a 8 different batches and can lead to high memeory usage. Maybe better system is required
# - train_loader
#     - train_loader_s1
#     - train_loader_s1
#     - train_loader_u1
#     - train_loader_u2
#     - adv_loader_s1
#     - adv_loader_s2
#     - adv_loader_u1
#     - adv_loader_u2

# In[12]:


manager, train_loader, val_loader = load_dataset(
    args.dataset,
    "aug4adv",
    dataset_root = args.dataset_root,
    supervised_ratio = args.supervised_ratio,
    batch_size = args.batch_size,
    train_folds = args.train_folds,
    val_folds = args.val_folds,
    
    augment_name_m1 = args.augment_m1,
    augment_name_m2 = args.augment_m2,
    verbose = 1
)


# ## Models

# In[45]:


torch.cuda.empty_cache()
model_func = get_model_from_name(args.model)

m1, m2 = model_func(manager=manager), model_func(manager=manager)

m1 = m1.cuda()
m2 = m2.cuda()


# ## training parameters

# In[46]:


# tensorboard
tensorboard_title = "%s_%s_%.1f_%s-%s" % (get_datetime(), model_func.__name__, args.supervised_ratio, args.augment_m1, args.augment_m2)
checkpoint_title = "%s_%.1f" % (model_func.__name__, args.supervised_ratio)
tensorboard = SummaryWriter(log_dir="%s/%s" % (args.tensorboard_path, tensorboard_title), comment=model_func.__name__)

# Losses
# see losses.py

# Optimizer
params = list(m1.parameters()) + list(m2.parameters())
optimizer = torch.optim.Adam(params, lr=args.learning_rate)

# define the warmups
lambda_cot = Warmup(args.lambda_cot_max, args.warmup_length, sigmoid_rampup)
lambda_diff = Warmup(args.lambda_diff_max, args.warmup_length, sigmoid_rampup)

# callback
lr_lambda = lambda epoch: (1.0 + np.cos((epoch-1) * np.pi / args.nb_epoch))
lr_scheduler = LambdaLR(optimizer, lr_lambda)
callbacks = [lr_scheduler, lambda_cot, lambda_diff]

# checkpoints
checkpoint_m1 = CheckPoint(m1, optimizer, mode="max", name="%s/%s_m1.torch" % (args.checkpoint_path, checkpoint_title))

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


# ## Can resume previous training
if args.resume:
    checkpoint_m1.load_last()

# ## Metrics and hyperparameters
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
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


# # Training functions
UNDERLINE_SEQ = "\033[1;4m"
RESET_SEQ = "\033[0m"

header_form = "{:<8.8} {:<6.6} - {:<6.6} - {:<8.8} {:<6.6} | {:<6.6} | {:<6.6} | {:<6.6} - {:<9.9} {:<9.9} | {:<9.9}- {:<6.6}"
value_form  = "{:<8.8} {:<6} - {:<6} - {:<8.8} {:<6.4f} | {:<6.4f} | {:<6.4f} | {:<6.4f} - {:<9.9} {:<9.4f} | {:<9.4f}- {:<6.4f}"

header = header_form.format(
    "", "Epoch", "%", "Losses:", "Lsup", "Lcot", "Ldiff", "total", "metrics: ", "acc_s1", "acc_u1","Time"
)


train_form = value_form
val_form = UNDERLINE_SEQ + value_form + RESET_SEQ

print(header)


# In[50]:


def split_to_cuda(x_y):
    x, y = x_y
    x = x.cuda()
    y = y.cuda()
    return x, y


# In[51]:

train_form = value_form
val_form = UNDERLINE_SEQ + value_form + RESET_SEQ

def format_data(data: tuple):
    def format_triple(data):
        S1, S2, U = data
        x_s1, y_s1 = S1
        x_s2, y_s2 = S2
        x_u, y_u = U

        x_s1, x_s2, x_u = x_s1.cuda().float(), x_s2.cuda().float(), x_u.cuda().float()
        y_s1, y_s2, y_u = y_s1.cuda().long(), y_s2.cuda().long(), y_u.cuda().long()
        return x_s1, x_s2, x_u, y_s1, y_s2, y_u
    
    def format_double(data: tuple):
        S, U = data
        x_s, _ = S
        x_u, _ = U
        x_s, x_u = x_s.cuda().float(), x_u.cuda().float()
        
        return x_s, x_u
    
    if len(data) == 3:
        return format_triple(data)
    elif len(data) == 2:
        return format_double(data)
    else:
        raise ValueError("data must a Iterable of size 2 or 3")

def train(epoch):
    start_time = time.time()
    print("")

    reset_metrics()
    m1.train()
    m2.train()

    for batch, (t_s1, t_s2, t_u1, t_u2, a_s1, a_s2, a_u1, a_u2) in enumerate(train_loader):
        x_s1, y_s1 = split_to_cuda(t_s1)
        x_s2, y_s2 = split_to_cuda(t_s2)
        x_u1, y_u1 = split_to_cuda(t_u1)
        x_u2, y_u2 = split_to_cuda(t_u2)
        
        ax_s1, ay_s1 = split_to_cuda(a_s1)
        ax_s2, ay_s2 = split_to_cuda(a_s2)
        ax_u1, ay_u1 = split_to_cuda(a_u1)
        ax_u2, ay_u2 = split_to_cuda(a_u2)

        # Predict normal data
        logits_s1 = m1(x_s1)
        logits_s2 = m2(x_s2)
        logits_u1 = m1(x_u1)
        logits_u2 = m2(x_u2)

        # pseudo labels of U
        pred_u1 = torch.argmax(logits_u1, 1)
        pred_u2 = torch.argmax(logits_u2, 1)
        
        # Predict augmented (adversarial data)
        adv_logits_s1 = m1(ax_s2)
        adv_logits_u1 = m1(ax_u2)
        adv_logits_s2 = m2(ax_s1)
        adv_logits_u2 = m2(ax_u1)

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

        total_loss = l_sup + lambda_cot() * l_cot + lambda_diff() * l_diff
        total_loss.backward()
        optimizer.step()

        # ======== Calc the metrics ========
        with torch.set_grad_enabled(False):
            # accuracies ----
            pred_s1 = torch.argmax(logits_s1, dim=1)
            pred_s2 = torch.argmax(logits_s2, dim=1)

            acc_s1 = metrics_fn["acc_s"][0](pred_s1, y_s1)
            acc_s2 = metrics_fn["acc_s"][1](pred_s2, y_s2)
            acc_u1 = metrics_fn["acc_u"][0](pred_u1, y_u1)
            acc_u2 = metrics_fn["acc_u"][1](pred_u2, y_u2)

            # ratios  ----
            adv_pred_s1 = torch.argmax(adv_logits_s1, 1)
            adv_pred_s2 = torch.argmax(adv_logits_s2, 1)
            adv_pred_u1 = torch.argmax(adv_logits_u1, 1)
            adv_pred_u2 = torch.argmax(adv_logits_u2, 1)

            ratio_s1 = metrics_fn["ratio_s"][0](adv_pred_s1, y_s1)
            ratio_s2 = metrics_fn["ratio_s"][1](adv_pred_s2, y_s2)
            ratio_u1 = metrics_fn["ratio_u"][0](adv_pred_u1, y_u1)
            ratio_u2 = metrics_fn["ratio_u"][1](adv_pred_u2, y_u2)
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
    tensorboard.add_scalar('train/Lsup', avg_sup.mean, epoch )
    tensorboard.add_scalar('train/Lcot', avg_cot.mean, epoch )
    tensorboard.add_scalar('train/Ldiff', avg_diff.mean, epoch )
    tensorboard.add_scalar("train/acc_1", acc_s1.mean, epoch )
    tensorboard.add_scalar("train/acc_2", acc_s2.mean, epoch )

    tensorboard.add_scalar("detail_acc/acc_s1", acc_s1.mean, epoch)
    tensorboard.add_scalar("detail_acc/acc_s2", acc_s2.mean, epoch)
    tensorboard.add_scalar("detail_acc/acc_u1", acc_u1.mean, epoch)
    tensorboard.add_scalar("detail_acc/acc_u2", acc_u2.mean, epoch)

    tensorboard.add_scalar("detail_ratio/ratio_s1", ratio_s1.mean, epoch)
    tensorboard.add_scalar("detail_ratio/ratio_s2", ratio_s2.mean, epoch)
    tensorboard.add_scalar("detail_ratio/ratio_u1", ratio_u1.mean, epoch)
    tensorboard.add_scalar("detail_ratio/ratio_u2", ratio_u2.mean, epoch)

    # Return the total loss to check for NaN
    return total_loss.item()


# In[52]:


def test(epoch, msg = ""):
    start_time = time.time()
    print("")

    reset_metrics()
    m1.eval()
    m2.eval()

    with torch.set_grad_enabled(False):
        for batch, (X, y) in enumerate(val_loader):
            x = X.cuda().float()
            y = y.cuda().long()

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
        
    tensorboard.add_scalar("max/acc_1", maximum_fn("acc_1", acc_1.mean), epoch )
    tensorboard.add_scalar("max/acc_2", maximum_fn("acc_2", acc_2.mean), epoch )
    
    tensorboard.add_scalar("detail_hyperparameters/lambda_cot", lambda_cot(), epoch)
    tensorboard.add_scalar("detail_hyperparameters/lambda_diff", lambda_diff(), epoch)
    tensorboard.add_scalar("detail_hyperparameters/learning_rate", get_lr(optimizer), epoch)

    # Apply callbacks
    for c in callbacks:
        c.step()

    # call checkpoint
    checkpoint_m1.step(acc_1.mean)


# In[ ]:


print(header)

for epoch in range(0, args.nb_epoch):
    total_loss = train(epoch)
    
    if np.isnan(total_loss):
        print("Losses are NaN, stoping the training here")
        break
        
    test(epoch)
    tensorboard.flush()
    
tensorboard.close()
