#!/usr/bin/env python
# coding: utf-8

# # import

# In[1]:


import os
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NU M_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
import numpy as np
import tqdm
import time

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from advertorch.attacks import GradientSignAttack
from torch.utils.tensorboard import SummaryWriter


# In[2]:


import sys
sys.path.append("../src/")

from datasetManager import DatasetManager
from generators import Generator
import signal_augmentations as sa 


# # Utils

# ## Metrics

# In[3]:


class Metrics:
    def __init__(self, epsilon=1e-10):
        self.value = 0
        self.accumulate_value = 0
        self.count = 0
        self.epsilon = epsilon
        
    def reset(self):
        self.accumulate_value = 0
        self.count = 0
        
    def __call__(self):
        self.count += 1

        
class BinaryAccuracy(Metrics):
    def __init__(self, epsilon=1e-10):
        Metrics.__init__(self, epsilon)
        
    def __call__(self, y_pred, y_true):
        super().__call__()
        
        with torch.set_grad_enabled(False):
            y_pred = (y_pred>0.5).float()
            correct = (y_pred == y_true).float().sum()
            self.value = correct/ (y_true.shape[0] * y_true.shape[1])
            
            self.accumulate_value += self.value
            return self.accumulate_value / self.count
        
        
class CategoricalAccuracy(Metrics):
    def __init__(self, epsilon=1e-10):
        Metrics.__init__(self, epsilon)
        
    def __call__(self, y_pred, y_true):
        super().__call__()
        
        with torch.set_grad_enabled(False):
            self.value = torch.mean((y_true == y_pred).float())
            self.accumulate_value += self.value

            return self.accumulate_value / self.count

        
class Ratio(Metrics):
    def __init__(self, epsilon=1e-10):
        Metrics.__init__(self, epsilon)
        
    def __call__(self, y_pred, y_adv_pred):
        super().__call__()
        
        results = zip(y_pred, y_adv_pred)
        results_bool = [int(r[0] != r[1]) for r in results]
        self.value = sum(results_bool) / len(results_bool) * 100
        self.accumulate_value += self.value
        
        return self.accumulate_value / self.count


# In[4]:


import datetime
def get_datetime():
    now = datetime.datetime.now()
    return str(now)[:10] + "_" + str(now)[11:-7]


# # Initialization

# ## set seeds

# In[5]:


def reset_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
reset_seed()


# ## Prepare GPU

# In[6]:


# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
# cudnn.benchmark = True


# # Model definition

# ## CNN
# https://arxiv.org/pdf/1608.04363.pdf

# In[7]:


class ConvPoolReLU(nn.Sequential):
    def __init__(self, in_size, out_size, kernel_size, stride, padding,
                pool_kernel_size, pool_stride):
        super(ConvPoolReLU, self).__init__(
            nn.Conv2d(in_size, out_size, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride),
            nn.BatchNorm2d(out_size),
            nn.ReLU6(inplace=True),
        )
        
class ConvReLU(nn.Sequential):
    def __init__(self, in_size, out_size, kernel_size, stride, padding):
        super(ConvReLU, self).__init__(
            nn.Conv2d(in_size, out_size, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU6(inplace=True),
        )


# In[8]:


class cnn(nn.Module):
    def __init__(self):
        super(cnn, self).__init__()
        
        self.features = nn.Sequential(
            ConvPoolReLU(1, 24, 3, 1, 1, (4,2), (4,2)),
            ConvPoolReLU(24, 48, 3, 1, 1, (4,2), (4,2)),
            ConvPoolReLU(48, 48, 3, 1, 1, (4,2), (4,2)),
            ConvReLU(48, 48, 3, 1, 1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(1008, 10),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.5),
#             nn.Linear(64, 10),
        )
                
        
    def forward(self, x):
        x = x.view(-1, 1, *x.shape[1:])

        x = self.features(x)
        x = self.classifier(x)
        
        return x


# # ======== Training ========

# ## Prep model

# In[10]:


torch.cuda.empty_cache()

# rnn
model_func = cnn
m1 = model_func()

m1.cuda()


# In[11]:


from torchsummaryX import summary
input_tensor = torch.zeros((1, 64, 173), dtype=torch.float)
input_tensor = input_tensor.cuda()

s = summary(m1, input_tensor)


# ## Prep data

# In[12]:


audio_root = "../dataset/audio"
metadata_root = "../dataset/metadata"

dataset = DatasetManager(metadata_root, audio_root, verbose=1)


# ## Prep training

# In[13]:


# create model
torch.cuda.empty_cache()

m1 = model_func()
m1.cuda()

# loss and optimizer
criterion_bce = nn.CrossEntropyLoss(reduction="mean")

optimizer = torch.optim.SGD(
    m1.parameters(),
    weight_decay=1e-3,
    lr=0.05
)

# Augmentation to use
# # ps1 = sa.PitchShift(0.5, DatasetManager.SR, (-2, 3))
# # ps2 = sa.PitchShift(0.5, DatasetManager.SR, (-3, 4))
# n = sa.Noise(0.5, (0.05, 0.2))
# augments = [ps1, ps2]
augments = []

# train and val loaders
train_dataset = Generator(dataset, augments=augments)

x, y = train_dataset.validation
x = torch.from_numpy(x)
y = torch.from_numpy(y)
val_dataset = torch.utils.data.TensorDataset(x, y)


# In[14]:


# training parameters
nb_epoch = 100
batch_size = 64
nb_batch = len(train_dataset) // batch_size

training_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# scheduler
lr_lambda = lambda epoch: 0.05 * (np.cos(np.pi * epoch / nb_epoch) + 1)
lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
callbacks = [lr_scheduler]
# callbacks = []

# tensorboard
title = "%s_%s_Cosd-lr_sgd-0.01lr-wd0.001_%de_no_augment" % ( get_datetime(), model_func.__name__, nb_epoch )
tensorboard = SummaryWriter(log_dir="tensorboard/%s" % title, comment=model_func.__name__)


# ## training

# In[ ]:


acc_func = CategoricalAccuracy()

for epoch in tqdm.tqdm(range(nb_epoch)):
    start_time = time.time()
    print("")
    
    acc_func.reset()

    m1.train()

    for i, (X, y) in enumerate(training_loader):        
        # Transfer to GPU
        X = X.cuda()
        y = y.cuda()
        
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
            X_val = X_val.cuda()
            y_val = y_val.cuda()

#             y_weak_val_pred, _ = model(X_val)
            logits = m1(X_val)
    # todo

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


# # ♫♪.ılılıll|̲̅̅●̲̅̅|̲̅̅=̲̅̅|̲̅̅●̲̅̅|llılılı.♫♪
