#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')


# # Import

# In[2]:


import os
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NU M_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
import time
import random
import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import torchvision.datasets
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from advertorch.attacks import GradientSignAttack


# In[3]:


from ubs8k.datasetManager import DatasetManager
from ubs8k.datasets import Dataset

from metric_utils.metrics import CategoricalAccuracy, FScore, ContinueAverage, Ratio
from DCT.util.checkpoint import CheckPoint
from DCT.util.utils import reset_seed, get_datetime, ZipCycle
from DCT.util.model_loader import get_model_from_name
from DCT.util.dataset_loader import load_dataset

from DCT.ramps import Warmup, sigmoid_rampup
# from DCT.losses import loss_cot, loss_diff, loss_sup


# # Arguments

# In[4]:


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--from_config", default="", type=str)
parser.add_argument("-d", "--dataset_root", default="../datasets", type=str)
parser.add_argument("-D", "--dataset", default="cifar10", type=str, help="available [ubs8k | cifar10]")

group_t = parser.add_argument_group("Commun parameters")
group_t.add_argument("--model", default="Pmodel", type=str)
group_t.add_argument("--supervised_ratio", default=0.08, type=float)
group_t.add_argument("--batch_size", default=200, type=int)
group_t.add_argument("--nb_epoch", default=600, type=int)
group_t.add_argument("--resume", action="store_true", default=False)
group_t.add_argument("--seed", default=1234, type=int)

group_o = parser.add_argument_group("Optimizer parameters")
group_o.add_argument("--learning_rate", default=0.05, type=int)
group_o.add_argument("--weight_decay", default=1e-4, type=float)
group_o.add_argument("--momentum", default=0.9, type=float)


group_u = parser.add_argument_group("UrbanSound8k parameters")
group_u.add_argument("-t", "--train_folds", nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8, 9], type=int)
group_u.add_argument("-v", "--val_folds", nargs="+", default=[10], type=int)

group_h = parser.add_argument_group('hyperparameters')
group_h.add_argument("--lambda_cot_max", default=10, type=float)
group_h.add_argument("--lambda_diff_max", default=0.5, type=float)
group_h.add_argument("--warmup_length", default=80, type=int)
group_h.add_argument("--epsilon", default=0.02, type=float)

group_a = parser.add_argument_group("Augmentation")
group_a.add_argument("--augment", action="append", help="augmentation. use as if python script")
group_a.add_argument("--augment_S", action="store_true", help="Apply augmentation on Supervised part")
group_a.add_argument("--augment_U", action="store_true", help="Apply augmentation on Unsupervised part")

group_l = parser.add_argument_group("Logs")
group_l.add_argument("--checkpoint_root", default="../model_save/", type=str)
group_l.add_argument("--tensorboard_root", default="../tensorboard/", type=str)
group_l.add_argument("--checkpoint_path", default="deep-co-training_ref", type=str)
group_l.add_argument("--tensorboard_path", default="deep-co-training_ref", type=str)
group_l.add_argument("--tensorboard_sufix", default="", type=str)

args = parser.parse_args("")

tensorboard_path = os.path.join(args.tensorboard_root, args.dataset, args.tensorboard_path)
checkpoint_path = os.path.join(args.checkpoint_root, args.dataset, args.checkpoint_path)


# # Initialization

# In[5]:


reset_seed(args.seed)


# # Prepare the dataset

# In[6]:


transform_train = transforms.Compose([
#     transforms.RandomAffine(0, translate=(1/16, 1/16)),
#     transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_val = transforms.Compose([
     transforms.ToTensor(),
])


# In[7]:


cifar10_path = os.path.join(args.dataset_root, "CIFAR10")


# In[8]:


train_dataset = torchvision.datasets.CIFAR10(root=cifar10_path, train=True, download=True, transform=transform_train)
val_dataset = torchvision.datasets.CIFAR10(root=cifar10_path, train=False, download=True, transform=transform_val)


# ## Split the dataset into Supervised and Unsupervised samples

# In[9]:


s_idx, u_idx = [], []
nb_s = int(np.ceil(len(train_dataset) * args.supervised_ratio) // 10)
cls_idx = [[] for _ in range(10)]


# In[10]:


for i in range(len(train_dataset)):
    _, y = train_dataset[i]
    cls_idx[y].append(i)
    
for i in range(len(cls_idx)):
    random.shuffle(cls_idx[i])
    
    s_idx += cls_idx[i][:nb_s]
    u_idx += cls_idx[i][nb_s:]


# # Prepare the training

# ## Create the model

# In[34]:


torch.cuda.empty_cache()
model_func = get_model_from_name(args.model)

m1 = model_func()
m2 = model_func()

m1 = m1.cuda()
m2 = m2.cuda()

m1 = torch.nn.DataParallel(m1)
m2 = torch.nn.DataParallel(m2)


# ## Calculate the S and U size of the batch

# In[35]:


s_batch_size = int(np.ceil(args.batch_size * args.supervised_ratio))
u_batch_size = args.batch_size - s_batch_size


# ## Create the data loader

# In[36]:


sampler_s = data.SubsetRandomSampler(s_idx)
sampler_u = data.SubsetRandomSampler(u_idx)

loader_s1 = data.DataLoader(train_dataset, batch_size=s_batch_size, sampler=sampler_s, num_workers=4)
loader_s2 = data.DataLoader(train_dataset, batch_size=s_batch_size, sampler=sampler_s, num_workers=4)
loader_u = data.DataLoader(train_dataset, batch_size=u_batch_size, sampler=sampler_u, num_workers=4)


# In[37]:


# train_loader = ZipCycle([loader_s1, loader_s2, loader_u])
val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)


# In[38]:


assert len(loader_s1) == len(loader_s2) == len(loader_u)


# ## Create the adversarial generator

# In[39]:


adv_generator_1 = GradientSignAttack(
    m1, loss_fn=nn.CrossEntropyLoss(reduction="sum"),
    eps=args.epsilon, clip_min=-np.inf, clip_max=np.inf, targeted=False
)

adv_generator_2 = GradientSignAttack(
    m2, loss_fn=nn.CrossEntropyLoss(reduction="sum"),
    eps=args.epsilon, clip_min=-np.inf, clip_max=np.inf, targeted=False
)


# ## Training parameters

# In[40]:


# tensorboard
tensorboard_title = "%s_%s_%.1fS" % (get_datetime(), model_func.__name__, args.supervised_ratio)
checkpoint_title = "%s_%.1fS" % (model_func.__name__, args.supervised_ratio)
tensorboard = SummaryWriter(log_dir="%s/%s" % (tensorboard_path, tensorboard_title), comment=model_func.__name__)
print(os.path.join(tensorboard_path, tensorboard_title))


# In[41]:


params = list(m1.parameters()) + list(m2.parameters())
optimizer = torch.optim.SGD(params, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

lr_lambda = lambda epoch: (1.0 + np.cos((epoch-1)*np.pi/args.nb_epoch)) * 0.5


# In[ ]:





# # =====================================

# In[42]:


# define the warmups
# lambda_cot = Warmup(args.lambda_cot_max, args.warmup_length, sigmoid_rampup)
# lambda_diff = Warmup(args.lambda_diff_max, args.warmup_length, sigmoid_rampup)
lambda_cot_max = args.lambda_cot_max
lambda_diff_max = args.lambda_diff_max
lambda_cot = 0.0
lambda_diff = 0.0

def adjust_learning_rate(optimizer, epoch):
    """cosine scheduling"""
    epoch = epoch + 1
    lr = args.learning_rate*(1.0 + np.cos((epoch-1)*np.pi/args.nb_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_lamda(epoch):
    epoch = epoch + 1
    global lambda_cot
    global lambda_diff
    if epoch <= args.warmup_length:
        lambda_cot = lambda_cot_max*np.exp(-5*(1-epoch/args.warmup_length)**2)
        lambda_diff = lambda_diff_max*np.exp(-5*(1-epoch/args.warmup_length)**2)
    else: 
        lambda_cot = lambda_cot_max
        lambda_diff = lambda_diff_max
        
# Define the losses
def loss_sup(logit_S1, logit_S2, labels_S1, labels_S2):
    ce = nn.CrossEntropyLoss() 
    loss1 = ce(logit_S1, labels_S1)
    loss2 = ce(logit_S2, labels_S2) 
    return (loss1+loss2)

def loss_cot(U_p1, U_p2):
# the Jensen-Shannon divergence between p1(x) and p2(x)
    S = nn.Softmax(dim = 1)
    LS = nn.LogSoftmax(dim = 1)
    a1 = 0.5 * (S(U_p1) + S(U_p2))
    loss1 = a1 * torch.log(a1)
    loss1 = -torch.sum(loss1)
    loss2 = S(U_p1) * LS(U_p1)
    loss2 = -torch.sum(loss2)
    loss3 = S(U_p2) * LS(U_p2)
    loss3 = -torch.sum(loss3)

    return (loss1 - 0.5 * (loss2 + loss3))/u_batch_size

def loss_diff(logit_S1, logit_S2, perturbed_logit_S1, perturbed_logit_S2, logit_U1, logit_U2, perturbed_logit_U1, perturbed_logit_U2):
    S = nn.Softmax(dim = 1)
    LS = nn.LogSoftmax(dim = 1)
    
    a = S(logit_S2) * LS(perturbed_logit_S1)
    a = torch.sum(a)

    b = S(logit_S1) * LS(perturbed_logit_S2)
    b = torch.sum(b)

    c = S(logit_U2) * LS(perturbed_logit_U1)
    c = torch.sum(c)

    d = S(logit_U1) * LS(perturbed_logit_U2)
    d = torch.sum(d)

    return -(a+b+c+d)/args.batch_size

# callback
# lr_scheduler = LambdaLR(optimizer, lr_lambda)
# callbacks = [lr_scheduler, lambda_cot, lambda_diff]
callbacks = []

# checkpoints
checkpoint_m1 = CheckPoint([m1, m2], optimizer, mode="max", name="%s/%s_m1.torch" % (checkpoint_path, checkpoint_title))

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


# In[43]:


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


# In[44]:


if args.resume:
    checkpoint_m1.load_last()


# In[ ]:





# # Training

# In[45]:


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


# In[46]:


def train(epoch):
    start_time = time.time()
    print("")

#     lambda_cot.step()
#     lambda_diff.step()

    adjust_learning_rate(optimizer, epoch)
    adjust_lamda(epoch)
    
    
    total_S1 = 0
    total_S2 = 0
    total_U1 = 0
    total_U2 = 0
    train_correct_S1 = 0
    train_correct_S2 = 0
    train_correct_U1 = 0
    train_correct_U2 = 0
    running_loss = 0.0
    ls = 0.0
    lc = 0.0 
    ld = 0.0
    
    reset_metrics()
    m1.train()
    m2.train()
    
    nb_batch = len(loader_s1)
    iter_s1 = iter(loader_s1)
    iter_s2 = iter(loader_s2)
    iter_u = iter(loader_u)

#     for batch, (S1, S2, U) in tqdm.tqdm_notebook(enumerate(train_loader)):
    for batch in tqdm.tqdm(range(nb_batch)):
#         x_s1, y_s1 = S1
#         x_s2, y_s2 = S2
#         x_u, y_u = U
        x_s1, y_s1 = iter_s1.next()
        x_s2, y_s2 = iter_s2.next()
        x_u, y_u = iter_u.next()

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

        #generate adversarial examples ----
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

#         total_loss = l_sup + lambda_cot() * l_cot + lambda_diff() * l_diff
        total_loss = l_sup + lambda_cot * l_cot + lambda_diff * l_diff
        total_loss.backward()
        optimizer.step()

        # ======== Calc the metrics ========
        with torch.set_grad_enabled(False):
            predictions_S1 = torch.argmax(logits_s1, dim=1)
            predictions_S2 = torch.argmax(logits_s2, dim=1)

            train_correct_S1 += torch.sum(predictions_S1 == y_s1)
            total_S1 += y_s1.size(0)

            train_correct_U1 += torch.sum(pred_u1 == y_u)
            total_U1 += y_u.size(0)

            train_correct_S2 += torch.sum(predictions_S2 == y_s2)
            total_S2 += y_s2.size(0)

            train_correct_U2 += torch.sum(pred_u2 == y_u)
            total_U2 += y_u.size(0)

            running_loss += total_loss.item()
            ls += l_sup.item()
            lc += l_cot.item()
            ld += l_diff.item()
            
            # using tensorboard to monitor loss and acc
            if (batch+1)%50 == 0:
                tensorboard.add_scalars('data/loss', {'loss_sup': l_sup.item(), 'loss_cot': l_cot.item(), 'loss_diff': l_diff.item()}, (epoch)*(nb_batch)+batch)
                tensorboard.add_scalars('data/training_accuracy', {'net1 acc': 100. * (train_correct_S1+train_correct_U1) / (total_S1+total_U1), 'net2 acc': 100. * (train_correct_S2+train_correct_U2) / (total_S2+total_U2)}, (epoch)*(nb_batch)+batch)
                # print statistics
                print('net1 training acc: %.3f%% | net2 training acc: %.3f%% | total loss: %.3f | loss_sup: %.3f | loss_cot: %.3f | loss_diff: %.3f  '
                    % (100. * (train_correct_S1+train_correct_U1) / (total_S1+total_U1), 100. * (train_correct_S2+train_correct_U2) / (total_S2+total_U2), running_loss/(batch+1), ls/(batch+1), lc/(batch+1), ld/(batch+1)))

    return total_loss.item()
        
            # accuracies ----
#             pred_s1 = torch.argmax(logits_s1, dim=1)
#             pred_s2 = torch.argmax(logits_s2, dim=1)

#             acc_s1 = metrics_fn["acc_s"][0](pred_s1, y_s1)
#             acc_s2 = metrics_fn["acc_s"][1](pred_s2, y_s2)
#             acc_u1 = metrics_fn["acc_u"][0](pred_u1, y_u)
#             acc_u2 = metrics_fn["acc_u"][1](pred_u2, y_u)

#             # ratios  ----
#             adv_pred_s1 = torch.argmax(adv_logits_s1, 1)
#             adv_pred_s2 = torch.argmax(adv_logits_s2, 1)
#             adv_pred_u1 = torch.argmax(adv_logits_u1, 1)
#             adv_pred_u2 = torch.argmax(adv_logits_u2, 1)

#             ratio_s1 = metrics_fn["ratio_s"][0](adv_pred_s1, y_s1)
#             ratio_s2 = metrics_fn["ratio_s"][0](adv_pred_s2, y_s2)
#             ratio_u1 = metrics_fn["ratio_s"][0](adv_pred_u1, y_u)
#             ratio_u2 = metrics_fn["ratio_s"][0](adv_pred_u2, y_u)
#             # ========

#             avg_total = metrics_fn["avg_total"](total_loss.item())
#             avg_sup = metrics_fn["avg_sup"](l_sup.item())
#             avg_diff = metrics_fn["avg_diff"](l_diff.item())
#             avg_cot = metrics_fn["avg_cot"](l_cot.item())

#             # logs
#             print(train_form.format(
#                 "Training: ",
#                 epoch + 1,
#                 int(100 * (batch + 1) / len(train_loader)),
#                 "", avg_sup.mean, avg_cot.mean, avg_diff.mean, avg_total.mean,
#                 "", acc_s1.mean, acc_u1.mean,
#                 time.time() - start_time
#             ), end="\r")


#     # using tensorboard to monitor loss and acc\n",
#     tensorboard.add_scalar('train/total_loss', avg_total.mean, epoch)
#     tensorboard.add_scalar('train/Lsup', avg_sup.mean, epoch )
#     tensorboard.add_scalar('train/Lcot', avg_cot.mean, epoch )
#     tensorboard.add_scalar('train/Ldiff', avg_diff.mean, epoch )
#     tensorboard.add_scalar("train/acc_1", acc_s1.mean, epoch )
#     tensorboard.add_scalar("train/acc_2", acc_s2.mean, epoch )

#     tensorboard.add_scalar("detail_acc/acc_s1", acc_s1.mean, epoch)
#     tensorboard.add_scalar("detail_acc/acc_s2", acc_s2.mean, epoch)
#     tensorboard.add_scalar("detail_acc/acc_u1", acc_u1.mean, epoch)
#     tensorboard.add_scalar("detail_acc/acc_u2", acc_u2.mean, epoch)

#     tensorboard.add_scalar("detail_ratio/ratio_s1", ratio_s1.mean, epoch)
#     tensorboard.add_scalar("detail_ratio/ratio_s2", ratio_s2.mean, epoch)
#     tensorboard.add_scalar("detail_ratio/ratio_u1", ratio_u1.mean, epoch)
#     tensorboard.add_scalar("detail_ratio/ratio_u2", ratio_u2.mean, epoch)

    # Return the total loss to check for NaN
#     return total_loss.item()


# In[47]:


def test(epoch, msg = ""):
    start_time = time.time()
    print("")

    reset_metrics()
    m1.eval()
    m2.eval()

    with torch.no_grad():
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
                int(100 * (batch + 1) / len(val_loader)),
                "", avg_sup.mean, 0.0, 0.0, avg_sup.mean,
                "", acc_1.mean, 0.0,
                time.time() - start_time
            ), end="\r")

    tensorboard.add_scalar("val/acc_1", acc_1.mean, epoch)
    tensorboard.add_scalar("val/acc_2", acc_2.mean, epoch)
    tensorboard.add_scalar("val/loss", l_sup.item(), epoch)
        
    tensorboard.add_scalar("max/acc_1", maximum_fn("acc_1", acc_1.mean), epoch )
    tensorboard.add_scalar("max/acc_2", maximum_fn("acc_2", acc_2.mean), epoch )
    
    tensorboard.add_scalar("detail_hyperparameters/lambda_cot", lambda_cot, epoch)
    tensorboard.add_scalar("detail_hyperparameters/lambda_diff", lambda_diff, epoch)
    tensorboard.add_scalar("detail_hyperparameters/learning_rate", get_lr(optimizer), epoch)

    # Apply callbacks
#     lr_scheduler.step()

    # call checkpoint
    checkpoint_m1.step([acc_1.mean, acc_2.mean])
    
    if epoch == 80:
        checkpoint_m1.save(".80")


# In[ ]:


print(header)

for epoch in range(0, 200):#args.nb_epoch):
    total_loss = train(epoch)
    
    if np.isnan(total_loss):
        print("Losses are NaN, stoping the training here")
        break
        
    test(epoch)

    tensorboard.flush()
    
tensorboard.close()


# In[ ]:





# In[ ]:




