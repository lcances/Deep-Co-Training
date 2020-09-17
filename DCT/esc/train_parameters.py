import torch.optim as optim
import numpy as np

def supervised(params):
    lr_lambda = lambda epoch: (1.0 + np.cos((epoch-1)*np.pi/args.nb_epoch)) * 0.5

    optimizers = optim.Adam(
        params = params,
        lr = 0.003,
    )
    
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    return optimizer, lr_scheduler


def dct(params):
    raise NotImplementedError


def dct_uniloss(params):
    raise NotImplementedError