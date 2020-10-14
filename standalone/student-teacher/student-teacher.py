#!/usr/bin/env python
# coding: utf-8

# # import
# https://arxiv.org/pdf/1703.01780.pdf
# In[1]:


#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')


# In[2]:


from torchsummary import summary
import argparse
from metric_utils.metrics import CategoricalAccuracy, FScore, ContinueAverage
from DCT.ramps import Warmup, sigmoid_rampup
from DCT.util.utils import reset_seed, get_datetime, track_maximum, dotdict
from DCT.util.checkpoint import CheckPoint, mSummaryWriter
from DCT.util.preprocess_loader import load_preprocesser
from DCT.util.model_loader import load_model
from DCT.util.callbacks_loader import load_callbacks
from DCT.util.optimizer_loader import load_optimizer
from DCT.util.dataset_loader import load_dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import autocast
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.utils.data as data
import torch.nn as nn
import torch
import numpy
import time
import os
import sys
sys.executable


# In[3]:


os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NU M_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"


# In[4]:


# # Arguments

# In[25]:


parser = argparse.ArgumentParser()
parser.add_argument("--from_config", default="", type=str)
parser.add_argument("-d", "--dataset_root", default="../../datasets", type=str)
parser.add_argument("-D", "--dataset", default="SpeechCommand", type=str)

group_t = parser.add_argument_group("Commun parameters")
group_t.add_argument("-m", "--model", default="cnn03", type=str)
group_t.add_argument("--supervised_ratio", default=0.1, type=float)
group_t.add_argument("--batch_size", default=256, type=int)
group_t.add_argument("--nb_epoch", default=200, type=int)
group_t.add_argument("--learning_rate", default=0.003, type=float)
group_t.add_argument("--resume", action="store_true", default=False)
group_t.add_argument("--seed", default=1234, type=int)

group_m = parser.add_argument_group("Model parameters")
group_m.add_argument("--num_classes", default=35, type=int)

group_u = parser.add_argument_group("Datasets parameters")
group_u.add_argument("-t", "--train_folds", nargs="+",
                     default=[1, 2, 3, 4], type=int)
group_u.add_argument("-v", "--val_folds", nargs="+", default=[5], type=int)

group_s = parser.add_argument_group("Student teacher parameters")
group_s.add_argument("--ema_alpha", default=0.999, type=float)
group_s.add_argument("--warmup_length", default=100, type=int)
group_s.add_argument("--lambda_cost_max", default=2, type=float)

group_l = parser.add_argument_group("Logs")
group_l.add_argument("--checkpoint_root",
                     default="../../model_save/", type=str)
group_l.add_argument("--tensorboard_root",
                     default="../../tensorboard/", type=str)
group_l.add_argument("--checkpoint_path", default="student-teacher", type=str)
group_l.add_argument("--tensorboard_path", default="student-teacher", type=str)
group_l.add_argument("--tensorboard_sufix", default="", type=str)

args = parser.parse_args()

tensorboard_path = os.path.join(
    args.tensorboard_root, args.dataset, args.tensorboard_path)
checkpoint_path = os.path.join(
    args.checkpoint_root, args.dataset, args.checkpoint_path)


# In[26]:


args


# # initialisation

# In[27]:


reset_seed(args.seed)


# # Prepare the dataset

# In[28]:


train_transform, val_transform = load_preprocesser(
    args.dataset, "student-teacher")
train_transform


# In[29]:


manager, train_loader, val_loader = load_dataset(
    args.dataset,
    "student-teacher",

    dataset_root=args.dataset_root,
    supervised_ratio=args.supervised_ratio,
    batch_size=args.batch_size,
    train_folds=args.train_folds,
    val_folds=args.val_folds,

    train_transform=train_transform,
    val_transform=val_transform,

    num_workers=0,
    pin_memory=True,

    verbose=2
)


# In[30]:


input_shape = tuple(train_loader._iterables[0].dataset[0][0].shape)
input_shape


# # Prep model

# In[31]:


torch.cuda.empty_cache()

model_func = load_model(args.dataset, args.model)

student = model_func(input_shape=input_shape, num_classes=args.num_classes)
teacher = model_func(input_shape=input_shape, num_classes=args.num_classes)

student = student.cuda()
teacher = teacher.cuda()

# We do not need gradient for the teacher model
for p in teacher.parameters():
    p.detach()


# In[32]:


s = summary(student, input_shape)


# # training parameters

# In[33]:


# tensorboard
title_element = (args.model, args.supervised_ratio, get_datetime(), model_func.__name__, args.supervised_ratio)
tensorboard_title = "%s/%sS/%s_%s_%.1fS" % title_element

title_element = (model_func.__name__, args.supervised_ratio)
checkpoint_title = "%s_%.1fS" % title_element

tensorboard = mSummaryWriter(log_dir="%s/%s" % (tensorboard_path, tensorboard_title), comment=model_func.__name__)
print(os.path.join(tensorboard_path, tensorboard_title))


# ## optimizer & callbacks

# In[34]:


optimizer = load_optimizer(args.dataset, "student-teacher", student=student, teacher=teacher, learning_rate=args.learning_rate)
callbacks = load_callbacks(args.dataset, "student-teacher", optimizer=optimizer, nb_epoch=args.nb_epoch)


# In[35]:


# losses
loss_ce = nn.CrossEntropyLoss(reduction="mean")  # Supervised loss
consistency_cost = nn.MSELoss(reduction="mean")  # Unsupervised loss

lambda_cost = Warmup(args.lambda_cost_max, args.warmup_length, sigmoid_rampup)
callbacks += [lambda_cost]

# Checkpoint
checkpoint = CheckPoint(student, optimizer, mode="max",
                        name="%s/%s.torch" % (checkpoint_path, checkpoint_title))


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


# ## Metrics definition

# In[36]:


def metrics_calculator():
    def c(logits, y):
        with torch.no_grad():
            y_one_hot = F.one_hot(y, num_classes=args.num_classes)

            pred = torch.softmax(logits, dim=1)
            arg = torch.argmax(logits, dim=1)

            acc = c.fn.acc(arg, y).mean
            f1 = c.fn.f1(pred, y_one_hot).mean

            return acc, f1,

    c.fn = dotdict(
        acc=CategoricalAccuracy(),
        f1=FScore(),
    )

    return c


# In[37]:


calc_student_s_metrics = metrics_calculator()
calc_student_u_metrics = metrics_calculator()
calc_teacher_s_metrics = metrics_calculator()
calc_teacher_u_metrics = metrics_calculator()

avg_Sce = ContinueAverage()
avg_Tce = ContinueAverage()
avg_ccost = ContinueAverage()

softmax_fn = nn.Softmax(dim=1)


def reset_metrics():
    for d in [calc_student_s_metrics.fn, calc_student_u_metrics.fn, calc_teacher_s_metrics.fn, calc_teacher_u_metrics.fn]:
        for fn in d.values():
            fn.reset()


maximum_tracker = track_maximum()


# ## Can resume previous training

# In[38]:


if args.resume:
    checkpoint.load_last()


# In[39]:


args.resume


# ## training function

# In[40]:


UNDERLINE_SEQ = "\033[1;4m"
RESET_SEQ = "\033[0m"

header_form = "{:<8.8} {:<6.6} - {:<6.6} - {:<10.8} {:<8.6} {:<8.6} {:<8.6} {:<8.6} {:<8.6} {:<8.6} | {:<10.8} {:<8.6} {:<8.6} {:<8.6} {:<8.6} {:<8.6} - {:<8.6}"
value_form = "{:<8.8} {:<6d} - {:<6d} - {:<10.8} {:<8.4f} {:<8.4f} {:<8.4f} {:<8.4f} {:<8.4f} {:<8.4f} | {:<10.8} {:<8.4f} {:<8.4f} {:<8.4f} {:<8.4f} {:<8.4f} - {:<8.4f}"
header = header_form.format(".               ", "Epoch",  "%", "Student:", "ce", "ccost",
                            "acc_s", "f1_s", "acc_u", "f1_u", "Teacher:", "ce", "acc_s", "f1_s", "acc_u", "f1_u", "Time")

train_form = value_form
val_form = UNDERLINE_SEQ + value_form + RESET_SEQ

print(header)


# In[41]:


def update_teacher_model(student_model, teacher_model, alpha, epoch):

    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (epoch + 1), alpha)

    for param, ema_param in zip(student_model.parameters(), teacher_model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data,  alpha=1-alpha)


# In[42]:


def train(epoch):
    start_time = time.time()
    print("")

    nb_batch = len(train_loader)

    reset_metrics()
    student.train()

    for i, (S, U) in enumerate(train_loader):
        x_s, y_s = S
        x_u, y_u = U

        x_s, x_u = x_s.cuda(), x_u.cuda()
        y_s, y_u = y_s.cuda(), y_u.cuda()

        # Predictions
        student_s_logits = student(x_s)
        student_u_logits = student(x_u)
        teacher_s_logits = teacher(x_s)
        teacher_u_logits = teacher(x_u)

        # Calculate supervised loss (only student on S)
        loss = loss_ce(student_s_logits, y_s)

        # Calculate consistency cost (mse(student(x), teacher(x))) x is S + U
        student_logits = torch.cat((student_s_logits, student_u_logits), dim=0)
        teacher_logits = torch.cat((teacher_s_logits, teacher_u_logits), dim=0)
        ccost = consistency_cost(softmax_fn(
            student_logits), softmax_fn(teacher_logits))

        total_loss = loss + lambda_cost() * ccost

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        with torch.set_grad_enabled(False):
            # Teacher prediction (for metrics purpose)
            _teacher_loss = loss_ce(teacher_s_logits, y_s)

            # Update teacher
            update_teacher_model(
                student, teacher, args.ema_alpha, epoch*nb_batch + i)

            # Compute the metrics for the student
            student_s_metrics = calc_student_s_metrics(student_s_logits, y_s)
            student_u_metrics = calc_student_u_metrics(student_u_logits, y_u)
            student_s_acc, student_u_acc, student_s_f1, student_u_f1 = * \
                student_s_metrics, *student_u_metrics

            # Compute the metrics for the teacher
            teacher_s_metrics = calc_teacher_s_metrics(teacher_s_logits, y_s)
            teacher_u_metrics = calc_teacher_u_metrics(teacher_u_logits, y_u)
            teacher_s_acc, teacher_u_acc, teacher_s_f1, teacher_u_f1 = * \
                teacher_s_metrics, *teacher_u_metrics

            # Running average of the two losses
            student_running_loss = avg_Sce(loss.item()).mean
            teacher_running_loss = avg_Tce(_teacher_loss.item()).mean
            running_ccost = avg_ccost(ccost.item()).mean

            # logs
            print(train_form.format(
                "Training: ", epoch + 1, int(100 * (i + 1) / nb_batch),
                "", student_running_loss, running_ccost, *
                student_s_metrics, *student_u_metrics,
                "", teacher_running_loss, *teacher_s_metrics, *teacher_u_metrics,
                time.time() - start_time
            ), end="\r")

    tensorboard.add_scalar("train/student_acc_s", student_s_acc, epoch)
    tensorboard.add_scalar("train/student_acc_u", student_u_acc, epoch)
    tensorboard.add_scalar("train/student_f1_s", student_s_f1, epoch)
    tensorboard.add_scalar("train/student_f1_u", student_u_f1, epoch)

    tensorboard.add_scalar("train/teacher_acc_s", teacher_s_acc, epoch)
    tensorboard.add_scalar("train/teacher_acc_u", teacher_u_acc, epoch)
    tensorboard.add_scalar("train/teacher_f1_s", teacher_s_f1, epoch)
    tensorboard.add_scalar("train/teacher_f1_u", teacher_u_f1, epoch)

    tensorboard.add_scalar("train/student_loss", student_running_loss, epoch)
    tensorboard.add_scalar("train/teacher_loss", teacher_running_loss, epoch)
    tensorboard.add_scalar("train/consistency_cost", running_ccost, epoch)


# In[43]:


def val(epoch):
    start_time = time.time()
    print("")
    reset_metrics()
    student.eval()

    with torch.set_grad_enabled(False):
        for i, (X, y) in enumerate(val_loader):
            X = X.cuda()
            y = y.cuda()

            # Predictions
            student_logits = student(X)
            teacher_logits = teacher(X)

            # Calculate supervised loss (only student on S)
            loss = loss_ce(student_logits, y)
            _teacher_loss = loss_ce(teacher_logits, y)  # for metrics only
            ccost = consistency_cost(softmax_fn(
                student_logits), softmax_fn(teacher_logits))

            # Compute the metrics
            y_one_hot = F.one_hot(y, num_classes=args.num_classes)

            # ---- student ----
            student_metrics = calc_student_s_metrics(student_logits, y)
            student_acc, student_f1 = student_metrics

            # ---- teacher ----
            teacher_metrics = calc_teacher_s_metrics(teacher_logits, y)
            teacher_acc, teacher_f1 = teacher_metrics

            # Running average of the two losses
            student_running_loss = avg_Sce(loss.item()).mean
            teacher_running_loss = avg_Tce(_teacher_loss.item()).mean
            running_ccost = avg_ccost(ccost.item()).mean

            # logs
            print(val_form.format(
                "Validation: ", epoch +
                1, int(100 * (i + 1) / len(val_loader)),
                "", student_running_loss, running_ccost, *student_metrics, 0.0, 0.0,
                "", teacher_running_loss, *teacher_metrics, 0.0, 0.0,
                time.time() - start_time
            ), end="\r")

    tensorboard.add_scalar("val/student_acc", student_acc, epoch)
    tensorboard.add_scalar("val/student_f1", student_f1, epoch)
    tensorboard.add_scalar("val/teacher_acc", teacher_acc, epoch)
    tensorboard.add_scalar("val/teacher_f1", teacher_f1, epoch)
    tensorboard.add_scalar("val/student_loss", student_running_loss, epoch)
    tensorboard.add_scalar("val/teacher_loss", teacher_running_loss, epoch)
    tensorboard.add_scalar("val/consistency_cost", running_ccost, epoch)

    tensorboard.add_scalar(
        "hyperparameters/learning_rate", get_lr(optimizer), epoch)
    tensorboard.add_scalar(
        "hyperparameters/lambda_cost_max", lambda_cost(), epoch)

    tensorboard.add_scalar(
        "max/student_acc", maximum_tracker("student_acc", student_acc), epoch)
    tensorboard.add_scalar(
        "max/teacher_acc", maximum_tracker("teacher_acc", teacher_acc), epoch)
    tensorboard.add_scalar(
        "max/student_f1", maximum_tracker("student_f1", student_f1), epoch)
    tensorboard.add_scalar(
        "max/teacher_f1", maximum_tracker("teacher_f1", teacher_f1), epoch)

    checkpoint.step(teacher_acc)
    for c in callbacks:
        c.step()


# # Training

# In[44]:


print(header)

start_epoch = checkpoint.epoch_counter
end_epoch = args.nb_epoch

for e in range(start_epoch, args.nb_epoch):
    train(e)
    val(e)

    tensorboard.flush()
tensorboard.close()


# ## Save the hyper parameters and the metrics

# In[25]:


hparams = {}
for key, value in args.__dict__.items():
    hparams[key] = str(value)

final_metrics = {
    "max_student_acc": maximum_tracker.max["student_acc"],
    "max_teacher_acc": maximum_tracker.max["teacher_acc"],
    "max_student_f1": maximum_tracker.max["student_f1"],
    "max_teacher_f1": maximum_tracker.max["teacher_f1"],
}

tensorboard.add_hparams(hparams, final_metrics)
