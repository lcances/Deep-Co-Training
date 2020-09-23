#!/usr/bin/env python
# coding: utf-8

# # import

# In[1]:


# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')


# In[2]:


from torchsummaryX import summary
import argparse
from metric_utils.metrics import CategoricalAccuracy, FScore, ContinueAverage
from DCT.util.utils import reset_seed, get_datetime, track_maximum
from DCT.util.checkpoint import CheckPoint
from DCT.util.preprocess_loader import load_preprocesser
from DCT.util.model_loader import load_model
from DCT.util.callbacks_loader import load_callbacks
from DCT.util.optimizer_loader import load_optimizer
from DCT.util.dataset_loader import load_dataset
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.nn as nn
import torch
import time
import os
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NU M_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"


# In[15]:


# # Arguments

# In[10]:


parser = argparse.ArgumentParser()
parser.add_argument("--from_config", default="", type=str)
parser.add_argument("-d", "--dataset_root", default="../../datasets", type=str)
parser.add_argument("-D", "--dataset", default="SpeechCommand",
                    type=str, help="available [ubs8k | cifar10]")

group_t = parser.add_argument_group("Commun parameters")
group_t.add_argument("-m", "--model", default="cnn03", type=str)
group_t.add_argument("--supervised_ratio", default=1.0, type=float)
group_t.add_argument("--batch_size", default=256, type=int)
group_t.add_argument("--nb_epoch", default=200, type=int)
group_t.add_argument("--learning_rate", default=0.001, type=float)
group_t.add_argument("--resume", action="store_true", default=False)
group_t.add_argument("--seed", default=1234, type=int)

group_m = parser.add_argument_group("Model parameters")
group_m.add_argument("--num_classes", default=50, type=int)

group_u = parser.add_argument_group("Datasets parameters")
group_u.add_argument("-t", "--train_folds", nargs="+",
                     default=[1, 2, 3, 4], type=int)
group_u.add_argument("-v", "--val_folds", nargs="+", default=[5], type=int)

group_l = parser.add_argument_group("Logs")
group_l.add_argument("--checkpoint_root",
                     default="../../model_save/", type=str)
group_l.add_argument("--tensorboard_root",
                     default="../../tensorboard/", type=str)
group_l.add_argument("--checkpoint_path", default="supervised", type=str)
group_l.add_argument("--tensorboard_path", default="supervised", type=str)
group_l.add_argument("--tensorboard_sufix", default="", type=str)

args = parser.parse_args()

tensorboard_path = os.path.join(
    args.tensorboard_root, args.dataset, args.tensorboard_path)
checkpoint_path = os.path.join(
    args.checkpoint_root, args.dataset, args.checkpoint_path)


if args.dataset not in ["ubs8k", "esc10", "esc50"]:
    print("Argument -t and -v only use for the following datasets"
          "[ubs8k, esc10, esc50]")

# In[11]:


args


# # initialisation

# In[12]:


reset_seed(args.seed)


# # Prepare the dataset

# In[13]:


args.dataset_root


# In[21]:


train_transform, val_transform = load_preprocesser(args.dataset, "supervised")


# In[25]:


manager, train_loader, val_loader = load_dataset(
    args.dataset,
    "supervised",

    dataset_root=args.dataset_root,
    supervised_ratio=args.supervised_ratio,
    batch_size=args.batch_size,
    train_folds=args.train_folds,
    val_folds=args.val_folds,

    train_transform=train_transform,
    val_transform=val_transform,

    verbose=2
)


# # Prep model

# In[29]:


torch.cuda.empty_cache()

model_func = load_model(args.dataset, args.model)
# model_func = get_model_from_name("esc_wideresnet28_8")
model = model_func(input_shape=(64, 32), num_classes=args.num_classes)
model = model.cuda()


# In[30]:


input_tensor = torch.zeros((1, 64, 32), dtype=torch.float)
input_tensor = input_tensor.cuda()

s = summary(model, input_tensor)


# ## Prep training

# In[31]:


nb_conv = 0

for layer in s.index.values:
    if "Conv" in layer:
        nb_conv += 1
print(nb_conv)


# # training parameters

# In[32]:


# tensorboard
title_element = (args.model, args.supervised_ratio,
                 get_datetime(), model_func.__name__, args.supervised_ratio)
tensorboard_title = "%s/%sS/%s_%s_%.1fS" % title_element

title_element = (model_func.__name__, args.supervised_ratio)
checkpoint_title = "%s_%.1fS" % title_element

tensorboard = SummaryWriter(
    log_dir=f"{tensorboard_path}/{tensorboard_title}",
    comment=model_func.__name__)
print(os.path.join(tensorboard_path, tensorboard_title))

# losses
loss_ce = nn.CrossEntropyLoss(reduction="mean")


# In[33]:


tensorboard_params = {}
for key, value in args.__dict__.items():
    tensorboard_params[key] = str(value)


# In[34]:


tensorboard.add_hparams(tensorboard_params, {})


# ## optimizer & callbacks

# In[37]:


optimizer = load_optimizer(args.dataset, "supervised", model=model)
callbacks = load_callbacks(args.dataset, "supervised",
                           optimizer=optimizer, nb_epoch=args.nb_epoch)


# In[42]:


# Checkpoint
checkpoint = CheckPoint(model, optimizer,
                        mode="max",
                        name=f"{checkpoint_path}/{checkpoint_title}.torch")

# Metrics
fscore_fn = FScore()
acc_fn = CategoricalAccuracy()
avg = ContinueAverage()
maximum_tracker = track_maximum()


def reset_metrics(): return [m.reset() for m in [fscore_fn, acc_fn, avg]]


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


# ## Can resume previous training

# In[49]:


if args.resume:
    checkpoint.load_last()


# In[50]:


args.resume


# ## training function

# In[51]:


UNDERLINE_SEQ = "\033[1;4m"
RESET_SEQ = "\033[0m"


header_form = "{:<8.8} {:<6.6} - {:<6.6} - {:<8.8} {:<6.6} - {:<9.9} {:<12.12}| {:<9.9}- {:<6.6}"
value_form = "{:<8.8} {:<6} - {:<6} - {:<8.8} {:<6.4f} - {:<9.9} {:<10.4f}| {:<9.4f}- {:<6.4f}"

header = header_form.format(
    ".               ", "Epoch", "%", "Losses:", "ce", "metrics: ", "acc", "F1 ", "Time"
)


train_form = value_form
val_form = UNDERLINE_SEQ + value_form + RESET_SEQ

print(header)


# In[52]:


def train(epoch):
    start_time = time.time()
    print("")

    reset_metrics()
    model.train()

    for i, (X, y) in enumerate(train_loader):
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
            y_one_hot = F.one_hot(y, num_classes=args.num_classes)

            acc = acc_fn(pred_arg, y).mean
            fscore = fscore_fn(pred, y_one_hot).mean
            avg_ce = avg(loss.item()).mean

            # logs
            print(train_form.format(
                "Training: ",
                epoch + 1,
                int(100 * (i + 1) / len(train_loader)),
                "", avg_ce,
                "", acc, fscore,
                time.time() - start_time
            ), end="\r")

    tensorboard.add_scalar("train/Lce", avg_ce, epoch)
    tensorboard.add_scalar("train/f1", fscore, epoch)
    tensorboard.add_scalar("train/acc", acc, epoch)


# In[53]:


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

            # metrics
            pred = torch.softmax(logits, dim=1)
            pred_arg = torch.argmax(logits, dim=1)
            y_one_hot = F.one_hot(y, num_classes=args.num_classes)

            acc = acc_fn(pred_arg, y).mean
            fscore = fscore_fn(pred, y_one_hot).mean
            avg_ce = avg(loss.item()).mean

            # logs
            print(val_form.format(
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

    tensorboard.add_scalar(
        "hyperparameters/learning_rate", get_lr(optimizer), epoch)

    tensorboard.add_scalar("max/acc", maximum_tracker("acc", acc), epoch)
    tensorboard.add_scalar("max/f1", maximum_tracker("f1", fscore), epoch)

    checkpoint.step(acc)
    for c in callbacks:
        c.step()


# In[54]:


print(header)

start_epoch = checkpoint.epoch_counter
end_epoch = args.nb_epoch

for e in range(start_epoch, args.nb_epoch):
    train(e)
    val(e)

    tensorboard.flush()
tensorboard.close()


# # ♫♪.ılılıll|̲̅̅●̲̅̅|̲̅̅=̲̅̅|̲̅̅●̲̅̅|llılılı.♫♪

# In[ ]:
