import numpy
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def get_lr_lambda(nb_epoch):
    def lr_lambda(epoch):
        return (1.0 + numpy.cos((epoch-1)*numpy.pi/nb_epoch)) * 0.5
    return lr_lambda


def supervised(nb_epoch: int, optimizer: Optimizer, **kwargs) -> list:
    lr_scheduler = LambdaLR(optimizer, get_lr_lambda(nb_epoch))
    return [lr_scheduler]


def dct(nb_epoch: int, optimizer: Optimizer, **kwargs) -> list:
    return supervised(nb_epoch, optimizer, **kwargs)


def dct_uniloss(nb_epoch: int, optimizer: Optimizer, **kwargs) -> list:
    return supervised(nb_epoch, optimizer, **kwargs)


def mean_teacher(nb_epoch: int, optimizer: Optimizer, **kwargs) -> list:
    return supervised(nb_epoch, optimizer, **kwargs)
