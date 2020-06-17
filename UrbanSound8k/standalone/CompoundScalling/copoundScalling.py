#!/usr/bin/env python
import os
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NU M_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
import numpy as np
import time
import gc

import librosa

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.append("../src/")

from UrbanSound8k.datasetManager import DatasetManager
from UrbanSound8k.generators import Dataset
from UrbanSound8k.utils import get_datetime, reset_seed
from UrbanSound8k.metrics import CategoricalAccuracy

from UrbanSound8k.datasetManager import conditional_cache

import argparse
parser = argparse.ArgumentParser(description='CopoundScalling')
parser.add_argument("-a", "--alpha", default=1.36, type=float, help="depth")
parser.add_argument("-b", "--beta", default=1.0, type=float, help="width")
parser.add_argument("-g", "--gamma", default=1.21, type=float, help="resolution")
parser.add_argument("-p", "--phi", default=1.0, type=float, help="Compound scaler")
parser.add_argument("-r", "--round_up", default=False, type=lambda x: (str(x).lower() in ['true','1', 'yes']), help="round up nb layer")
parser.add_argument("-t", "--title", default="default", type=str, help="Tensorboard base title")
args = parser.parse_args()

reset_seed(1324)


def conditional_cache(func):
    def decorator(*args, **kwargs):
        if "filename" in kwargs.keys() and "cached" in kwargs.keys():
            filename = kwargs["filename"]
            cached = kwargs["cached"]

            if filename is not None and cached:
                if filename not in decorator.cache.keys():
                    decorator.cache[filename] = func(*args, **kwargs)
                    return decorator.cache[filename]

                else:
                    if decorator.cache[filename] is None:
                        decorator.cache[filename] = func(*args, **kwargs)
                        return decorator.cache[filename]
                    else:
                        return decorator.cache[filename]

        return func(*args, **kwargs)

    decorator.cache = dict()

    return decorator


# In[6]:


class ConvBNReLUPool(nn.Sequential):
    def __init__(self, in_size, out_size, kernel_size, stride, padding,
                pool_kernel_size, pool_stride, dropout: float = 0.0):
        super(ConvBNReLUPool, self).__init__(
            nn.Conv2d(in_size, out_size, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_size),
            nn.Dropout2d(dropout),
            nn.ReLU6(inplace=True),
            nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride),
        )
        
class ConvReLU(nn.Sequential):
    def __init__(self, in_size, out_size, kernel_size, stride, padding):
        super(ConvReLU, self).__init__(
            nn.Conv2d(in_size, out_size, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU6(inplace=True),
        )


# In[7]:


class ScalableCnn(nn.Module):
    """
    Compound Scaling based CNN
    see: https://arxiv.org/pdf/1905.11946.pdf
    """

    def __init__(self, dataset: DatasetManager,
                 compound_scales: tuple = (1, 1, 1),
                 initial_conv_inputs=[1, 32, 64, 64],
                 initial_conv_outputs=[32, 64, 64, 64],
                 initial_linear_inputs=[1344, ],
                 initial_linear_outputs=[10, ],
                 initial_resolution=[64, 173],
                 round_up: bool = False,
                 **kwargs
                 ):
        super(ScalableCnn, self).__init__()
        self.compound_scales = compound_scales
        self.dataset = dataset
        round_func = np.floor if not round_up else np.ceil
        print("round_func: ", round_func)

        alpha, beta, gamma = compound_scales[0], compound_scales[1], compound_scales[2]

        initial_nb_conv = len(initial_conv_inputs)
        initial_nb_dense = len(initial_linear_inputs)

        # Apply compound scaling

        # resolution ----
        # WARNING - RESOLUTION WILL CHANGE THE FEATURES EXTRACTION OF THE SAMPLE
        new_n_mels = int(round_func(initial_resolution[0] * gamma))
        new_n_time_bins = int(round_func(initial_resolution[1] * gamma))
        new_hop_length = int(round_func( (self.dataset.sr * DatasetManager.LENGTH) / new_n_time_bins))

        self.scaled_resolution = (new_n_mels, new_n_time_bins)
        print("new scaled resolution: ", self.scaled_resolution)

        dataset.extract_feature = self.generate_feature_extractor(new_n_mels, new_hop_length)

        # ======== CONVOLUTION PARTS ========
        # ---- depth ----
        scaled_nb_conv = round_func(initial_nb_conv * alpha)
        
        new_conv_inputs, new_conv_outputs = initial_conv_inputs.copy(), initial_conv_outputs.copy()
        if scaled_nb_conv != initial_nb_conv:  # Another conv layer must be created
            print("More conv layer must be created")
            gaps = np.array(initial_conv_outputs) - np.array(initial_conv_inputs)  # average filter gap
            avg_gap = gaps.mean()

            while len(new_conv_inputs) < scaled_nb_conv:
                new_conv_outputs.append(int(round_func(new_conv_outputs[-1] + avg_gap)))
                new_conv_inputs.append(new_conv_outputs[-2])
        
        # ---- width ----
        scaled_conv_inputs = [int(round_func(i * beta)) for i in new_conv_inputs]
        scaled_conv_outputs = [int(round_func(i * beta)) for i in new_conv_outputs]
        
        print("new conv layers:")
        print("inputs: ", scaled_conv_inputs)
        print("ouputs: ", scaled_conv_outputs)
        
        # Check how many conv with pooling layer can be used
        nb_max_pooling = int(np.floor(np.min([np.log2(self.scaled_resolution[0]), int(np.log2(self.scaled_resolution[1]))])))
        nb_model_pooling = len(scaled_conv_inputs)

        if nb_model_pooling > nb_max_pooling:
            nb_model_pooling = nb_max_pooling
            
        # fixe initial conv layers
        scaled_conv_inputs[0] = 1
        
        # ======== LINEAR PARTS ========
        # adjust the first dense input with the last convolutional layers
        initial_linear_inputs[0] = self.calc_initial_dense_input(
            self.scaled_resolution,
            nb_model_pooling,
            scaled_conv_outputs
        )
        
        # --- depth ---
        scaled_nb_linear = round_func(initial_nb_dense * alpha)
        
        if scaled_nb_linear != initial_nb_dense:  # Another dense layer must be created
            print("More dense layer must be created")
            dense_list = np.linspace(initial_linear_inputs[0], initial_linear_outputs[-1], scaled_nb_linear + 1)
            initial_linear_inputs = dense_list[:-1]
            initial_linear_outputs = dense_list[1:]
            
        # --- width ---
        scaled_dense_inputs = [int(round_func(i * beta)) for i in initial_linear_inputs]
        scaled_dense_outputs = [int(round_func(i * beta)) for i in initial_linear_outputs]
        
        # fix first and final linear layer
        scaled_dense_inputs[0] = self.calc_initial_dense_input(self.scaled_resolution,
                                                                nb_model_pooling,
                                                                scaled_conv_outputs)
        scaled_dense_outputs[-1] = 10
        
        print("new dense layers:")
        print("inputs: ", scaled_dense_inputs)
        print("ouputs: ", scaled_dense_outputs)

        # ======== BUILD THE MODEL=========
        # features part ----
        features = []

        # Create the layers
        for idx, (inp, out) in enumerate(zip(scaled_conv_inputs, scaled_conv_outputs)):
            if idx < nb_model_pooling:
                dropout = 0.3 if idx != 0 else 0.0
                features.append(ConvBNReLUPool(inp, out, 3, 1, 1, (2, 2), (2, 2), dropout))

            else:
                features.append(ConvReLU(inp, out, 3, 1, 1))

        self.features = nn.Sequential(
            *features,
        )

        # classifier part ----
        linears = []
        for inp, out in zip(scaled_dense_inputs[:-1], scaled_dense_outputs[:-1]):
            print(inp, out)
            linears.append(nn.Linear(inp, out))
            linears.append(nn.ReLU6(inplace=True))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            *linears,
            nn.Linear(scaled_dense_inputs[-1], scaled_dense_outputs[-1])
        )

    def forward(self, x):
        x = x.view(-1, 1, *x.shape[1:])

        x = self.features(x)
        x = self.classifier(x)

        return x

    def calc_initial_dense_input(self, resolution, nb_model_pooling, conv_outputs):
        dim1 = resolution[0]
        dim2 = resolution[1]

        for i in range(int(nb_model_pooling)):
            dim1 = dim1 // 2
            dim2 = dim2 // 2

        return dim1 * dim2 * conv_outputs[-1]

    def generate_feature_extractor(self, n_mels, hop_length):
        @conditional_cache
        def extract_feature(raw_data, filename = None, cached = False):
            feat = librosa.feature.melspectrogram(
                raw_data, self.dataset.sr, n_fft=2048, hop_length=hop_length, n_mels=n_mels, fmin=0, fmax=self.dataset.sr // 2)
            feat = librosa.power_to_db(feat, ref=np.max)
            return feat

        return extract_feature


# create model
phi = args.phi
alpha = args.alpha**phi
beta = args.beta**phi
gamma = args.gamma**phi

torch.cuda.empty_cache()

# Baseline
parameters = dict(
    #dataset=manager,
    
    compound_scales = (alpha, beta, gamma),
    
    initial_conv_inputs=[1, 32, 64, 64],
    initial_conv_outputs=[32, 64, 64, 64],
    initial_linear_inputs=[1344, ],
    initial_linear_outputs=[10, ],
    initial_resolution=[64, 173],
    round_up=args.round_up,
)

# Step 1 (alpha = 1.36, beta=1.0, gamma=1.21, phi=1.0)
# parameters = dict(
#     #dataset=manager,
    
#     compound_scales = compound_scales,
#     initial_conv_inputs=[1, 32, 64, 64, 64, 80],
#     initial_conv_outputs=[32, 64, 64, 64, 80, 96],
#     initial_linear_inputs=[1344,  677],
#     initial_linear_outputs=[677,  10],
#     initial_resolution=[78, 210],
#     round_up=True,
# )



# # Prep training

# In[10]:


class Trainer:
    def __init__(self, audio_root, metadata_root,
        model_parameters, criterion,
        batch_size=64, nb_epoch=100, augmentations = []
    ):
        self.audio_root = audio_root
        self.metadata_root = metadata_root
        
        self.model_parameters = model_parameters
        self.criterion = criterion
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        
        self.model = None
        self.manager = None
        self.tensorboard = None
        self.train_dataset = None
        self.val_dataset = None
        self.train_loader = None
        self.val_loader = None
        
        self.data_ready: bool = False
        self.training_ready: bool = False
            
    def _free(self):
        self.model = None
        self.manager = None
        self.tensorboard = None
        self.train_dataset = None
        self.val_dataset = None
        self.train_loader = None
        self.val_loader = None
        
        gc.collect()
        
    def prepare_data(self, train_fold: list = [1,2,3,4,5,6,7,8,9], val_fold: list = [10]):

        self.manager = DatasetManager(
            self.metadata_root, self.audio_root,
            train_fold=train_fold, val_fold = val_fold,
            verbose=1)
        
        # train and val loaders
        self.train_dataset = Dataset(self.manager, train=True, val=False, augments=[], cached=True)
        self.val_dataset = Dataset(self.manager, train=False, val=True, augments=[], cached=True)
        
        self.training_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True)
        
        self.nb_batch = len(self.train_dataset) // self.batch_size
        
        self.data_ready = True
        
    def prepare_training(self, tensorboard_title: str = None, run_id: int = None):
            
        # Create model
        self.model = ScalableCnn(dataset=self.manager, **self.model_parameters)
        self.model.cuda()
        
        if tensorboard_title is not None:
            if run_id is not None:
                run_id_ = "%s_run%d" % (get_datetime(), (run_id + 1))
                self.tensorboard = SummaryWriter(log_dir="../../tensorboard/compound_scaling/%s/%.3fa_%.3fb_%.3fg_%.3fp/%s" % (tensorboard_title, alpha, beta, gamma, phi, run_id_), comment=self.model.__class__.__name__)
                
            else:
                self.tensorboard = SummaryWriter(log_dir="../../tensorboard/compound_scaling/%s/%.3fa_%.3fb_%.3fg_%.3fp/%s" % (tensorboard_title, alpha, beta, gamma, phi, get_datetime()), comment=self.model.__class__.__name__)

        self.optimizer = torch.optim.SGD(self.model.parameters(), weight_decay=1e-3, lr=0.05)
        
        # scheduler
        lr_lambda = lambda epoch: 0.05 * (np.cos(np.pi * epoch / self.nb_epoch) + 1)
        lr_scheduler = LambdaLR(self.optimizer, lr_lambda=lr_lambda)
        self.callbacks = [lr_scheduler]
        
        # metrics
        self.acc_func = CategoricalAccuracy()
        
        self.training_ready = True
        
    def reset_metrics(self):
        self.acc_func.reset()
        
    def cross_val(self, tensorboard_title: str = None):
        train_folds = [
            [2, 3, 4, 5, 6, 7, 8, 9, 10], 
            [1, 3, 4, 5, 6, 7, 8, 9, 10], 
            [1, 2, 4, 5, 6, 7, 8, 9, 10],
            [1, 2, 3, 5, 6, 7, 8, 9, 10],
            [1, 2, 3, 4, 6, 7, 8, 9, 10],
            [1, 2, 3, 4, 5, 7, 8, 9, 10],
            [1, 2, 3, 4, 5, 6, 8, 9, 10],
            [1, 2, 3, 4, 5, 6, 7, 9, 10],
            [1, 2, 3, 4, 5, 6, 7, 8, 10],
            [1, 2, 3, 4, 5, 6, 7, 8, 9], 
        ]
        
        val_folds = [[1], [2], [3], [4], [5], [6], [7] ,[8], [9], [10]]
        
        for run_id in range(10):
                       
            self._free() # Free all memory
            self.prepare_data(train_fold=train_folds[run_id], val_fold=val_folds[run_id])
            self.prepare_training(tensorboard_title=tensorboard_title, run_id=run_id)
            
            self.train()
        
    def train(self):
        if not self.data_ready:
            raise ValueError("The data are not ready, please use `prepare_data(..)`")
        if not self.training_ready:
            raise ValueError("The training is not ready, please use `prepare_training(...)`")
            
        for epoch in range(self.nb_epoch):
            self.train_step(epoch)
            self.val_step(epoch)
            
            for callback in self.callbacks:
                callback.step()
                
        if self.tensorboard is not None:
            self.tensorboard.flush()
            self.tensorboard.close()
        
    def train_step(self, epoch):
        start_time = time.time()
        print("")

        self.reset_metrics()
        self.model.train()

        for i, (X, y) in enumerate(self.training_loader):        
            # Transfer to GPU
            X = X.cuda()
            y = y.cuda()

            # predict
            logits = self.model(X)

            weak_loss = self.criterion(logits, y)

            total_loss = weak_loss

            # calc metrics
    #         y_pred = torch.log_softmax(logits, dim=1)
            _, y_pred = torch.max(logits, 1)
            acc = self.acc_func(y_pred, y)

            # ======== back propagation ========
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            # ======== history ========
            print("Epoch {}, {:d}% \t ce: {:.4f} - acc: {:.4f} - took: {:.2f}s".format(
                epoch+1,
                int(100 * (i+1) / self.nb_batch),
                total_loss.item(),
                acc,
                time.time() - start_time
            ),end="\r")

        # using tensorboard to monitor loss and acc
        if self.tensorboard is not None:
            self.tensorboard.add_scalar('train/ce', total_loss.item(), epoch)
            self.tensorboard.add_scalar("train/acc", 100. * acc, epoch )
            
    def val_step(self, epoch):
        print("")
        with torch.set_grad_enabled(False):
            # reset metrics
            self.reset_metrics()
            self.model.eval()

            for i, (X_val, y_val) in enumerate(self.val_loader):
                # Transfer to GPU
                X_val = X_val.cuda()
                y_val = y_val.cuda()

    #             y_weak_val_pred, _ = model(X_val)
                logits = self.model(X_val)

                # calc loss
                weak_loss_val = self.criterion(logits, y_val)

                # metrics
    #             y_val_pred =torch.log_softmax(logits, dim=1)
                _, y_val_pred = torch.max(logits, 1)
                acc_val = self.acc_func(y_val_pred, y_val)

                #Print statistics
                print("Epoch {}, {:d}% \t ce val: {:.4f} - acc val: {:.4f}".format(
                    epoch+1,
                    int(100 * (i+1) / self.nb_batch),
                    weak_loss_val.item(),
                    acc_val,
                ),end="\r")

            # using tensorboard to monitor loss and acc
            if self.tensorboard is not None:
                self.tensorboard.add_scalar('validation/ce', weak_loss_val.item(), epoch)
                self.tensorboard.add_scalar("validation/acc", 100. * acc_val, epoch )


# In[11]:


criterion = nn.CrossEntropyLoss(reduce="mean")


torch.cuda.empty_cache()

trainer = Trainer(
    audio_root="../../dataset/audio", metadata_root="../../dataset/metadata",
    model_parameters = parameters,
    criterion=criterion,
    batch_size=64,
    nb_epoch=150,
)

title = "%s/cross_validation" % args.title

trainer.cross_val(title)
