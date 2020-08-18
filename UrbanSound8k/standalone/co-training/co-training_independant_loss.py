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

import sys
sys.path.append("../../..")

from util.utils import reset_seed, get_datetime, get_model_from_name, ZipCycle
from metric_utils.metrics import CategoricalAccuracy, FScore, ContinueAverage, Ratio
from util.checkpoint import CheckPoint

from UrbanSound8k.ramps import Warmup, sigmoid_rampup, sigmoid_rampdown
from UrbanSound8k.losses import loss_cot, loss_diff, loss_sup


# # Arguments

# In[548]:


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset_root", default="../../../datasets/ubs8k", type=str)
# parser.add_argument("--supervised_mult", default=1.0, type=float)

group_t = parser.add_argument_group("Training parameters")
group_t.add_argument("--supervised_ratio", default=0.1, type=float)
group_t.add_argument("-t", "--train_folds", nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8, 9], type=int)
group_t.add_argument("-v", "--val_folds", nargs="+", default=[10], type=int)
group_t.add_argument("--model", default="cnn03", type=str)
group_t.add_argument("--batch_size", default=100, type=int)
group_t.add_argument("--nb_epoch", default=1000, type=int)
group_t.add_argument("--learning_rate", default=0.0005, type=float)
group_t.add_argument("--resume", action="store_true", default=False)
group_t.add_argument("--seed", default=1234, type=int)

group_h = parser.add_argument_group('hyperparameters')
group_h.add_argument("--lambda_cot_max", default=10, type=float)
group_h.add_argument("--lambda_diff_max", default=0.5, type=float)
group_h.add_argument("--warmup_length", default=1000, type=int)
group_h.add_argument("--epsilon", default=0.02, type=float)

group = parser.add_argument_group('uniloss')
group.add_argument("--loss_scheduler", default="no_rule", type=str)
group.add_argument("--steps", default=1, type=int)
group.add_argument("--cycle", default=1, type=int)
group.add_argument("--beta", default=1, type=int)
group.add_argument("--plsup_mini", default=0.0, type=float)

group_a = parser.add_argument_group("Augmentation")
group_a.add_argument("--augment", action="append", help="augmentation. use as if python script")
group_a.add_argument("--augment_S", action="store_true", help="Apply augmentation on Supervised part")
group_a.add_argument("--augment_U", action="store_true", help="Apply augmentation on Unsupervised part")

group_l = parser.add_argument_group("Logs")
group_l.add_argument("--checkpoint_path", default="../../model_save/deep-co-training_independant-loss", type=str)
group_l.add_argument("--tensorboard_path", default="../../tensorboard/deep-co-training_independant-loss", type=str)
group_l.add_argument("--tensorboard_sufix", default="", type=str)

args = parser.parse_args()


reset_seed(args.seed)

# ## Prepare the dataset

# In[5]:


audio_root = os.path.join(args.dataset_root, "audio")
metadata_root = os.path.join(args.dataset_root, "metadata")
all_folds = args.train_folds + args.val_folds

manager = DatasetManager(
    metadata_root, audio_root,
    folds=all_folds,
    verbose=1
)


# In[24]:


# prepare the sampler with the specified number of supervised file
train_dataset = Dataset(manager, folds=args.train_folds, cached=True)
val_dataset = Dataset(manager, folds=args.val_folds, cached=True)


# ## Models

# In[549]:


torch.cuda.empty_cache()
model_func = get_model_from_name(args.model)

m1, m2 = model_func(manager=manager), model_func(manager=manager)

m1 = m1.cuda()
m2 = m2.cuda()


# ## Adversarial generator

# In[550]:


# adversarial generation
adv_generator_1 = GradientSignAttack(
    m1, loss_fn=nn.CrossEntropyLoss(reduction="sum"),
    eps=args.epsilon, clip_min=-80, clip_max=0, targeted=True
)

adv_generator_2 = GradientSignAttack(
    m2, loss_fn=nn.CrossEntropyLoss(reduction="sum"),
    eps=args.epsilon, clip_min=-80, clip_max=0, targeted=True
)


# ## Rules makers

# In[551]:


def uniform_rule(steps: int = 0, **kwargs):
    sup_steps = [0.34 for _ in range(steps)]
    cot_steps = [0.33 for _ in range(steps)]
    diff_steps = [0.33 for _ in range(steps)]

    return sup_steps, cot_steps, diff_steps

def weighted_uniform_rule(steps: int = 0, **kwargs):
    lcm = args.lambda_cot_max
    ldm = args.lambda_diff_max
    lsm = 1
    total = lcm + ldm + lsm

    sup_steps = [lsm * 0.34 for _ in range(steps)]
    cot_steps = [lcm * 0.33 for _ in range(steps)]
    diff_steps = [ldm * 0.33 for _ in range(steps)]

    # normalize
    for i in range(steps):
        summed = sup_steps[i] + cot_steps[i] + diff_steps[i]
        sup_steps[i] /= summed
        cot_steps[i] /= summed
        diff_steps[i] /= summed
        
    return sup_steps, cot_steps, diff_steps

def weighted_linear_rule(step: int = 10, plsup_mini: float = 0.0, **kwargs):
    lcm = args.lambda_cot_max
    ldm = args.lambda_diff_max
    lsm = 1
    print("inner plsup_mini: ", plsup_mini)
    
    s_start, s_end = 1, lsm * (plsup_mini)
    c_start, c_end = 0, lcm * (0.5 - (plsup_mini / 2))
    d_start, d_end = 0, ldm * (0.5 - (plsup_mini / 2))
    
    # normalize
    total = s_end + c_end + d_end
    s_end /= total
    c_end /= total
    d_end /= total
    
    sup_steps = np.linspace(s_start, s_end, step)
    cot_steps = np.linspace(d_start, c_end, step)
    diff_steps = np.linspace(d_start, d_end, step)
    
    return sup_steps, cot_steps, diff_steps

def linear_rule(step: int = 10, plsup_mini: float = 0.0, **kwargs):
    hop_length = np.linspace(0, args.nb_epoch, step)
    
    s_start, s_end = 1, plsup_mini
    c_start, c_end = 0, 0.5 - (plsup_mini / 2)
    d_start, d_end = 0, 0.5 - (plsup_mini / 2)
    
    sup_steps = np.linspace(s_start, s_end, step)
    cot_steps = np.linspace(d_start, c_end, step)
    diff_steps = np.linspace(d_start, d_end, step)

     # normalize
    for i in range(step):
        summed = sup_steps[i] + cot_steps[i] + diff_steps[i]
        if summed != 1:
            sup_steps[i] /= summed
            cot_steps[i] /= summed
            diff_steps[i] /= summed
    
    return sup_steps, cot_steps, diff_steps

def cosine_rule(step: int = 10, cycle: int = 1, **kwargs):
    hop_length = np.linspace(0, args.nb_epoch, step * cycle)
    
    sup_steps = 0.5 * (np.cos(np.pi * hop_length / (args.nb_epoch / cycle)) + 1)
    cot_steps = 1 - (0.5 * (np.cos(np.pi * hop_length / (args.nb_epoch / cycle)) + 1))
    diff_steps =  1 - (0.5 * (np.cos(np.pi * hop_length / (args.nb_epoch / cycle)) + 1))
    
    # normalize
    for i in range(step * cycle):
        summed = sup_steps[i] + cot_steps[i] + diff_steps[i]
        sup_steps[i] /= summed
        cot_steps[i] /= summed
        diff_steps[i] /= summed
        
    return sup_steps, cot_steps, diff_steps

def weighted_cosine_rule(step: int = 10, cycle: int = 1, **kwargs):
    lcm = args.lambda_cot_max
    ldm = args.lambda_diff_max
    lsm = 1
    
    hop_length = np.linspace(0, args.nb_epoch, step * cycle)
    
    sup_steps = 0.5 * (np.cos(np.pi * hop_length / (args.nb_epoch / cycle)) + 1)
    cot_steps = 1 - (0.5 * (np.cos(np.pi * hop_length / (args.nb_epoch / cycle)) + 1))
    diff_steps =  1 - (0.5 * (np.cos(np.pi * hop_length / (args.nb_epoch / cycle)) + 1))
    
    sup_steps *= lsm
    cot_steps *= lcm
    diff_steps *= ldm
    
    # normalize
    for i in range(step * cycle):
        summed = sup_steps[i] + cot_steps[i] + diff_steps[i]
        sup_steps[i] /= summed
        cot_steps[i] /= summed
        diff_steps[i] /= summed
        
    return sup_steps, cot_steps, diff_steps

def annealing_cosine_rule(step: int = 10, cycle: int = 1, beta: float = 2, **kwargs):
    hop_length = np.linspace(0, args.nb_epoch, step * cycle)
    
    # create original steps
    decaying =  np.exp(-beta * np.linspace(0, 1, len(hop_length)))
    
    sup_steps = 0.5 * (decaying * np.cos(np.pi * hop_length / (args.nb_epoch / cycle)) + 1)
    cot_steps = 1 - (0.5 * (decaying * np.cos(np.pi * hop_length / (args.nb_epoch / cycle)) + 1))
    diff_steps =  1 - (0.5 * (decaying * np.cos(np.pi * hop_length / (args.nb_epoch / cycle)) + 1))
    
    # normalize
    for i in range(step * cycle):
        summed = sup_steps[i] + cot_steps[i] + diff_steps[i]
        sup_steps[i] /= summed
        cot_steps[i] /= summed
        diff_steps[i] /= summed
        
    return sup_steps, cot_steps, diff_steps

def weighted_annealing_cosine_rule(step: int = 10, cycle: int = 1, beta: float = 2, **kwargs):
    lcm = args.lambda_cot_max
    ldm = args.lambda_diff_max
    lsm = 1
    
    hop_length = np.linspace(0, args.nb_epoch, step * cycle)
    
    # create original steps
    decaying =  np.exp(-beta * np.linspace(0, 1, len(hop_length)))
    
    sup_steps = 0.5 * (decaying * np.cos(np.pi * hop_length / (args.nb_epoch / cycle)) + 1)
    cot_steps = 1 - (0.5 * (decaying * np.cos(np.pi * hop_length / (args.nb_epoch / cycle)) + 1))
    diff_steps =  1 - (0.5 * (decaying * np.cos(np.pi * hop_length / (args.nb_epoch / cycle)) + 1))
    
    sup_steps *= lsm
    cot_steps *= lcm
    diff_steps *= ldm
    
    # normalize
    for i in range(step * cycle):
        summed = sup_steps[i] + cot_steps[i] + diff_steps[i]
        sup_steps[i] /= summed
        cot_steps[i] /= summed
        diff_steps[i] /= summed
        
    return sup_steps, cot_steps, diff_steps

def sigmoid_rule(steps: int = 10, **kwargs):
    hop_length = np.linspace(0, args.nb_epoch, steps)
    
    sup_steps = np.asarray([sigmoid_rampdown(x, args.nb_epoch) for x in hop_length])
    cot_steps = np.asarray([sigmoid_rampup(x, args.nb_epoch) for x in hop_length])
    diff_steps = np.asarray([sigmoid_rampup(x, args.nb_epoch) for x in hop_length])
    
    # normalize
    for i in range(steps):
        summed = sup_steps[i] + cot_steps[i] + diff_steps[i]
        sup_steps[i] /= summed
        cot_steps[i] /= summed
        diff_steps[i] /= summed

    return sup_steps, cot_steps, diff_steps
    

def weighted_sigmoid_rule(steps: int = 10, **kwargs):
    lcm = args.lambda_cot_max
    ldm = args.lambda_diff_max
    lsm = 1

    hop_length = np.linspace(0, args.nb_epoch, steps)
    
    sup_steps = np.asarray([lsm * sigmoid_rampdown(x, args.nb_epoch) for x in hop_length])
    cot_steps = np.asarray([lcm * sigmoid_rampup(x, args.nb_epoch) for x in hop_length])
    diff_steps = np.asarray([ldm * sigmoid_rampup(x, args.nb_epoch) for x in hop_length])
    
    # normalize
    for i in range(steps):
        summed = sup_steps[i] + cot_steps[i] + diff_steps[i]
        sup_steps[i] /= summed
        cot_steps[i] /= summed
        diff_steps[i] /= summed

    return sup_steps, cot_steps, diff_steps
    

def rule_maker(rule_fn, steps: int = 5, **kwargs):
    p_lsup, p_lcot, p_ldiff = rule_fn(steps, **kwargs)
    
    # making the rules
    hop_length = np.linspace(0, args.nb_epoch, steps)

    rules = dict()
    for i, epoch in enumerate(hop_length):
        rules[epoch] = {"lsup": p_lsup[i], "lcot": p_lcot[i], "ldiff": p_ldiff[i]}
        
    return rules

def loss_chooser(epoch):
    for k in reversed(rules.keys()):
        if epoch >= k:
            chance = list(rules[k].values())
            break
    
    loss_function = ["sup", "cot", "diff"]
    return np.random.choice(loss_function, p=chance)

def rule_chooser(rule_str):
    mapper = {
        "uniform": uniform_rule,
        "weighted-uniform": weighted_uniform_rule,
        "linear": linear_rule,
        "weighted-linear": weighted_linear_rule,
        "sigmoid": sigmoid_rule,
        "weighted-sigmoid": weighted_sigmoid_rule,
        "cosine": cosine_rule,
        "weighted-cosine": weighted_cosine_rule,
        "annealing-cosine": annealing_cosine_rule,
        "weighted-annealing-cosine": weighted_annealing_cosine_rule,
    }
    
    if rule_str not in mapper:
     raise ValueError("Loss scheduler must be from [%s]" % (", ".join(mapper.keys())))
        
    return mapper[rule_str]

rule_fn = rule_chooser(args.loss_scheduler)
loss_scheduler_args = {
    "steps": args.steps,
    "cycle": args.cycle,
    "beta": args.beta,
    "plsup_mini": args.plsup_mini,
}

rules = rule_maker(rule_fn, **loss_scheduler_args)

# ## Loaders

# In[553]:


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


# ## training parameters

# In[554]:


# tensorboard
tensorboard_title = "%s_%s_%.1fS_%se_%slr_%s-rfn_%ss_%s" % (get_datetime(), model_func.__name__, args.supervised_ratio, args.nb_epoch, args.learning_rate, args.loss_scheduler, args.steps, args.tensorboard_sufix)
checkpoint_title = "%s_%.1fS_%se_%slr_%s-rfn_%ss_%s" % (model_func.__name__, args.supervised_ratio, args.nb_epoch, args.learning_rate, args.loss_scheduler, args.steps, args.tensorboard_sufix)
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


# In[555]:


reset_metrics()


# ## Can resume previous training

# In[556]:


args.resume


# In[557]:


if args.resume:
    checkpoint_m1.load_last()


# ## Metrics and hyperparameters

# In[558]:


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

# In[559]:


UNDERLINE_SEQ = "\033[1;4m"

RESET_SEQ = "\033[0m"


header_form = "{:<8.8} {:<6.6} - {:<6.6} - {:<8.8} {:<6.6} | {:<6.6} | {:<6.6} | {:<6.6} - {:<9.9} {:<9.9} | {:<9.9}- {:<6.6}"
value_form  = "{:<8.8} {:<6} - {:<6} - {:<8.8} {:<6.4f} | {:<6.4f} | {:<6.4f} | {:<6.4f} - {:<9.9} {:<9.4f} | {:<9.4f}- {:<6.4f}"

header = header_form.format(
    "", "Epoch", "%", "Losses:", "Lsup", "Lcot", "Ldiff", "total", "metrics: ", "acc_s1", "acc_u1","Time"
)


train_form = value_form
val_form = UNDERLINE_SEQ + value_form + RESET_SEQ

def mprint(epoch, msg, end):
    if epoch % 10 == 0:
        print(msg, end=end)
        
print(header)


# ## Train sup

# In[560]:


def train_sup_helper(start_time, epoch, batch, x_s1, x_s2, y_s1, y_s2):
    x_s1, x_s2, = x_s1.cuda(), x_s2.cuda()
    y_s1, y_s2, = y_s1.cuda(), y_s2.cuda()

    logits_s1 = m1(x_s1)
    logits_s2 = m2(x_s2)

    l_sup = loss_sup(logits_s1, logits_s2, y_s1, y_s2)
    total_loss = l_sup
    
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    # Calc the metrics
    with torch.set_grad_enabled(False):

        # accuracies ----
        pred_s1 = torch.argmax(logits_s1, dim=1)
        pred_s2 = torch.argmax(logits_s2, dim=1)

        acc_s1 = metrics_fn["acc_s"][0](pred_s1, y_s1)
        acc_s2 = metrics_fn["acc_s"][1](pred_s2, y_s2)

        avg_total = metrics_fn["avg_total"](total_loss.item())
        avg_sup = metrics_fn["avg_sup"](l_sup.item())

        # logs
        mprint(epoch, train_form.format(
            "Training: ",
            epoch + 1,
            int(100 * (batch + 1) / len(train_loader)),
            "", avg_sup.mean, 0.0, 0.0, avg_total.mean,
            "", acc_s1.mean, 0.0,
            time.time() - start_time
        ), end="\r")

    # using tensorboard to monitor loss and acc\n",
    tensorboard.add_scalar('train/total_loss', avg_total.mean, epoch)
    tensorboard.add_scalar('train/Lsup', avg_sup.mean, epoch )
    tensorboard.add_scalar("train/acc_1", acc_s1.mean, epoch )
    tensorboard.add_scalar("train/acc_2", acc_s2.mean, epoch )

    tensorboard.add_scalar("detail_acc/acc_s1", acc_s1.mean, epoch)
    tensorboard.add_scalar("detail_acc/acc_s2", acc_s2.mean, epoch)

    # Return the total loss to check for NaN
    return total_loss.item()
    


# ## Train cot

# In[561]:


def train_cot_helper(start_time, epoch, batch, x_u, y_u):
    x_u = x_u.cuda()
    y_u = y_u.cuda()

    logits_u1 = m1(x_u)
    logits_u2 = m2(x_u)

    l_cot = loss_cot(logits_u1, logits_u2)
    total_loss = l_cot * args.lambda_cot_max

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    # ======== Calc the metrics ========
    with torch.set_grad_enabled(False):
        pred_u1 = torch.argmax(logits_u1, 1)
        pred_u2 = torch.argmax(logits_u2, 1)
        acc_u1 = metrics_fn["acc_u"][0](pred_u1, y_u)
        acc_u2 = metrics_fn["acc_u"][1](pred_u2, y_u)

        avg_total = metrics_fn["avg_total"](total_loss.item())
        avg_cot = metrics_fn["avg_cot"](l_cot.item())

        # logs
        mprint(epoch, train_form.format(
            "Training: ",
            epoch + 1,
            int(100 * (batch + 1) / len(train_loader)),
            "", 0.0, avg_cot.mean, 0.0, avg_total.mean,
            "", 0.0, acc_u1.mean,
            time.time() - start_time
        ), end="\r")


    # using tensorboard to monitor loss and acc\n",
    tensorboard.add_scalar('train/total_loss', avg_total.mean, epoch)
    tensorboard.add_scalar('train/Lcot', avg_cot.mean, epoch )

    tensorboard.add_scalar("detail_acc/acc_u1", acc_u1.mean, epoch)
    tensorboard.add_scalar("detail_acc/acc_u2", acc_u2.mean, epoch)

    # Return the total loss to check for NaN
    return total_loss.item()


# ## Train diff

# In[562]:


def train_diff_helper(start_time, epoch, batch, x_s1, x_s2, x_u, y_s1, y_s2, y_u):
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

    l_diff = loss_diff(
        logits_s1, logits_s2, adv_logits_s1, adv_logits_s2,
        logits_u1, logits_u2, adv_logits_u1, adv_logits_u2
    )
    total_loss = l_diff * args.lambda_diff_max

    optimizer.zero_grad()
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
        ratio_s2 = metrics_fn["ratio_s"][1](adv_pred_s2, y_s2)
        ratio_u1 = metrics_fn["ratio_u"][0](adv_pred_u1, y_u)
        ratio_u2 = metrics_fn["ratio_u"][1](adv_pred_u2, y_u)
        # ========

        avg_total = metrics_fn["avg_total"](total_loss.item())
        avg_diff = metrics_fn["avg_diff"](l_diff.item())

        # logs
        mprint(epoch, train_form.format(
            "Training: ",
            epoch + 1,
            int(100 * (batch + 1) / len(train_loader)),
            "", 0.0, 0.0, avg_diff.mean, avg_total.mean,
            "", acc_s1.mean, acc_u1.mean,
            time.time() - start_time
        ), end="\r")


    # using tensorboard to monitor loss and acc\n",
    tensorboard.add_scalar('train/total_loss', avg_total.mean, epoch)
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


# In[563]:


def train(epoch):
    st = time.time()
    mprint(epoch, "", "\n")

    reset_metrics()
    m1.train()
    m2.train()
    
    for batch, (S1, S2, U) in enumerate(train_loader):
        chosen_loss = loss_chooser(epoch)
        
        x_s1, y_s1 = S1
        x_s2, y_s2 = S2
        x_u, y_u = U

        if chosen_loss == "sup":
            return train_sup_helper(st, epoch, batch, x_s1, x_s2, y_s1, y_s2)
        
        elif chosen_loss == "cot":
            return train_cot_helper(st, epoch, batch, x_u, y_u)
        
        elif chosen_loss == "diff":
            return train_diff_helper(st, epoch, batch, x_s1, x_s2, x_u, y_s1, y_s2, y_u)


# In[564]:


def test(epoch, msg = ""):
    start_time = time.time()
    mprint(epoch, "", "\n")

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
            mprint(epoch, val_form.format(
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


# # Training

# In[565]:


print(header)

for epoch in range(0, args.nb_epoch):
    total_loss = train(epoch)
    
    if np.isnan(total_loss):
        print("Losses are NaN, stoping the training here")
        break
        
    test(epoch)

tensorboard.flush()
tensorboard.close()


# In[566]:


print("rule function: ", rule_fn)
print("model: ", model_func)
print("max acc 1: %.2f" % (maximum_fn.max["acc_1"].item() * 100))
print("max acc 2: %.2f" % (maximum_fn.max["acc_2"].item() * 100))


# # ♫♪.ılılıll|̲̅̅●̲̅̅|̲̅̅=̲̅̅|̲̅̅●̲̅̅|llılılı.♫♪
