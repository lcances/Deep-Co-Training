#!/usr/bin/env python
# coding: utf-8

# # import

# In[2]:


import os
import sys
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NU M_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
import numpy as np
import tqdm
import time
import random

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from advertorch.attacks import GradientSignAttack
from torch.utils.tensorboard import SummaryWriter


# In[3]:


from ubs8k.datasetManager import DatasetManager
from ubs8k.generators import Dataset
import ubs8k.signal_augmentations as sa 
from ubs8k.models import ScalableCnn
import ubs8k.metrics as metrics
from ubs8k.utils import reset_seed, get_datetime

import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument("-t", "--train_folds", nargs="+", default="1 2 3 4 5 6 7 8 9", required=True, type=int, help="fold to use for training")
parser.add_argument("-v", "--val_folds", nargs="+", default="10", type=int, required=True, help="fold to use for validation")
parser.add_argument("-a", "--alpha", default=1, type=float)
parser.add_argument("-b", "--beta", default=1, type=float)
parser.add_argument("-g", "--gamma", default=1, type=float)
args = parser.parse_args()
# # set seeds

# In[4]:


reset_seed(1324)


audio_root = "../../dataset/audio"
metadata_root = "../../dataset/metadata"

manager = DatasetManager(metadata_root, audio_root, verbose=1)


#  # Prep scalling factors

# In[6]:



# # Prep training

# In[8]:


class CopoundScaling:
    def __init__(self, nb_epochs, batch_size, criterion, augments = [], metrics = [], tensorboard_dir="copoundScaling"):
        self.model = None
        self.manager = None
        self.nb_epochs = nb_epochs
        self.batch_size = batch_size
        self.optimizer = None
        self.criterion=criterion
        self.metrics_to_apply = metrics
        self.metrics_name = [m.__class__.__name__ for m in metrics]
        self.callbacks = []
        self.tensorboard_dir = tensorboard_dir
        self.tensorboard = None
        self.run = 0

    def set_model(self, new_model):
        self.model = new_model
        self._init_dataset(self.model.dataset) # TODO <-- raname dataset <> manager
        
        self.run += 1 # count number of run done with the trainer. Usefull when doing cross validation
        
    def _init_dataset(self, manager):
        self.manager = manager
        
         # Prepare dataset
        self.train_dataset = Dataset(manager, train=True, val=False, augments=augments, cached=True)
        self.val_dataset = Dataset(manager, train=False, val=True, augments=augments, cached=True)
        self.nb_batch = len(self.train_dataset) // batch_size
        
        # prepare loaders
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=batch_size, shuffle=True)
        
    def set_optimizer(self, new_optimizer):
        self.optimizer = new_optimizer
        
    def set_callbacks(self, new_callbacks):
        self.callbacks = new_callbacks
        
    def init_tensorboard(self):
        title = self.model.__class__.__name__
        date_time = get_datetime()
        (a, b, g) = self.model.compound_scales
        abg = "%.4f_%.4f_%.4f" % (a, b, g)
        return SummaryWriter("tensorboard/%s/%s/%s_run%s_%s" % (self.tensorboard_dir, abg, date_time, self.run, title))
        
    def learn(self):
        if self.model is None:
            raise ValueError("A model must be define, please use set_model()")
        
        torch.cuda.empty_cache()
        self.tensorboard = self.init_tensorboard()
        
        for e in range(self.nb_epochs):
            self.train(e)
            self.val(e)
            
            for c in self.callbacks:
                c.step()

        self.tensorboard.flush()
        self.tensorboard.close()
 
    def train(self, epoch):
        start_time = time.time()
        print("")

        self._reset_metrics()
        self.model.train()

        for i, (X, y) in enumerate(self.train_loader):        
            # Transfer to GPU
            X = X.cuda()
            y = y.cuda()

            # predict
            logits = self.model(X)

            loss = self.criterion(logits, y)

            # calc metrics
            _, y_pred = torch.max(logits, 1)
            metric_t_values = [func(y_pred, y) for func in self.metrics_to_apply]
            metric_values = [t.cpu().numpy() for t in metric_t_values]

            # ======== back propagation ========
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # ======== history ========
            msg = "Epoch {}, {:d}% \t ce: {:.4f} - metrics: " + ", ".join(["{:.4f}" for _ in range(len(metric_values))]) + "- took: {:.2f}s"
            print(msg.format(epoch+1, int(100 * (i+1) / self.nb_batch), loss.item(), *metric_values, time.time() - start_time), end="\r")

        # using tensorboard to monitor loss and acc
        self.tensorboard.add_scalar('train/ce', loss.item(), epoch)
        for metric_name, metric_value in zip(self.metrics_name, metric_values):
            self.tensorboard.add_scalar("train/%s" % metric_name, metric_value, epoch)
    
    def val(self, epoch):
        # Validation
        with torch.set_grad_enabled(False):
            
            print("")
            self._reset_metrics()
            self.model.eval()

            for X_val, y_val in self.val_loader:
                # Transfer to GPU
                X_val = X_val.cuda()
                y_val = y_val.cuda()

                logits = self.model(X_val)

                # calc loss
                loss_val = self.criterion(logits, y_val)

                # metrics
                _, y_val_pred = torch.max(logits, 1)
                metric_t_values = [func(y_val_pred, y_val) for func in self.metrics_to_apply]
                metric_values = [t.cpu().numpy() for t in metric_t_values]

                #Print statistics
                msg = "validation metric_values: " + ", ".join(["{:.4f}" for _ in range(len(metric_values))])
                print(msg.format(*metric_values), end='\r')

            # using tensorboard to monitor loss and acc
            self.tensorboard.add_scalar('validation/ce', loss_val.item(), epoch)
            for metric_name, metric_value in zip(self.metrics_name, metric_values):
                self.tensorboard.add_scalar("validation/%s" % metric_name, metric_value, epoch)
    
    def _reset_metrics(self):
        for m in self.metrics_to_apply:
            m.reset()


# # Prep parameters

# In[9]:


# create model
torch.cuda.empty_cache()

nb_epoch = 100
batch_size = 64

# loss and optimizer
criterion_bce = nn.CrossEntropyLoss(reduction="mean")

# Augmentation to use
augments = []

acc_func = metrics.CategoricalAccuracy()
trainer = CopoundScaling(nb_epoch, batch_size, criterion_bce, augments, metrics=[acc_func])


# Training

torch.cuda.empty_cache()
    
manager = DatasetManager(metadata_root, audio_root, train_fold=args.train_folds, val_fold=args.val_folds, verbose=1)
a = args.alpha
b = args.beta
g = args.gamma

print("=============================")
print("alpha = %f, beta = %f, gamma = %f" % (a, b, g))
print("=============================")

model_func = ScalableCnn
model = model_func(
    manager, 
    compound_scales=(a, b, g),
    initial_conv_inputs=[1, 32, 64, 64],
    initial_conv_outputs=[32, 64, 64, 64],
    initial_linear_inputs=[1344, ],
    initial_linear_outputs=[10, ],
    initial_resolution=[64, 173],
    round_up = True
)
optimizer = torch.optim.SGD(
    model.parameters(),
    weight_decay=1e-3,
    lr=0.04
)

# scheduler depend on the optimizer
lr_lambda = lambda epoch: (1.0 + np.cos((epoch-1)*np.pi/nb_epoch))
lr_scheduler = LambdaLR(optimizer, lr_lambda)

model.cuda()
trainer.set_model(model)
trainer.set_optimizer(optimizer)
trainer.set_callbacks([lr_scheduler])

trainer.learn()
