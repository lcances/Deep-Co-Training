import torch.optim as optim
import numpy as np

def supervised(params):
    def lr_lambda(e):

        if e < 60: return 1
        elif 60 <= e < 120: return 0.2
        elif 120 <= e < 160: return 0.04
        else: return 0.008

    optimizers = optim.SGD(
        params = params,
        lr = 0.1,
        momentum=0.9
        weight_decay=0.0005,
    )
    
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    return optimizer, lr_scheduler


def dct(params):
    optimizer = optim.SGD(
        params=params,
        lr=0.1,
        momentum=0.9,
        weight_decay=0.0001,
    )

    lr_lambda = lambda epoch: (1.0 + np.cos((epoch-1)*np.pi/args.nb_epoch)) * 0.5
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    return optimizer, lr_scheduler


def dct_uniloss(params):
    optimizers = optim.SGD(
        params = params,
        lr = 0.1,
        momentum=0.9,
        weight_decay=0.0001,
    )

    lr_lambda = lambda epoch: (1.0 + np.cos((epoch-1)*np.pi/args.nb_epoch)) * 0.5
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    return optimizer, lr_scheduler