import numpy
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def supervised(nb_epoch: int, optimizer: Optimizer, **kwargs) -> list:
    def lr_lambda(epoch):
        return (1.0 + numpy.cos((epoch-1)*numpy.pi/nb_epoch)) * 0.5

    lr_scheduler = LambdaLR(optimizer, lr_lambda)
    return [lr_scheduler]


def dct(nb_epoch: int, optimizer: Optimizer, **kwargs) -> list:
    def lr_lambda(epoch):
        return (1.0 + numpy.cos((epoch-1)*numpy.pi/nb_epoch)) * 0.5

    lr_scheduler = LambdaLR(optimizer, lr_lambda)
    return [lr_scheduler]


def dct_uniloss(nb_epoch: int, optimizer: Optimizer, **kwargs) -> list:
    def lr_lambda(epoch):
        return (1.0 + numpy.cos((epoch-1)*numpy.pi/nb_epoch)) * 0.5

    lr_scheduler = LambdaLR(optimizer, lr_lambda)
    return [lr_scheduler]