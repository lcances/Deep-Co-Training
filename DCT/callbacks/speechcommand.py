
import numpy


def supervised(nb_epoch: int) -> list:
    def lr_lambda(epoch):
        return (1.0 + numpy.cos((epoch-1)*numpy.pi/nb_epoch)) * 0.5

    return [lr_lambda]
