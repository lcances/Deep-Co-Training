import torch
import torch.nn as nn


class Sequential_adv(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class ConvPoolReLU(nn.Sequential):
    def __init__(self, in_size, out_size, kernel_size, stride, padding,
                 pool_kernel_size, pool_stride):
        super(ConvPoolReLU, self).__init__(
            nn.Conv2d(in_size, out_size, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride),
            nn.BatchNorm2d(out_size),
            nn.ReLU6(inplace=True),
        )

        
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


class ConvAdvBNReLUPool(nn.Module):
    def __init__(self, in_size, out_size, kernel_size, stride, padding,
                 pool_kernel_size, pool_stride, dropout: float = 0.0):
        super().__init__()

        self.conv = nn.Conv2d(in_size, out_size, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn_normal = nn.BatchNorm2d(out_size)
        self.bn_adv = nn.BatchNorm2d(out_size)

        self.final = nn.Sequential(
            nn.Dropout2d(dropout),
            nn.ReLU6(inplace=True),
            nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride),
        )

    def forward(self, x, adv: bool = False):
        x = self.conv(x)

        x = self.bn_adv(x) if adv else self.bn_normal(x)

        return self.final(x)


class ConvReLU(nn.Sequential):
    def __init__(self, in_size, out_size, kernel_size, stride, padding):
        super(ConvReLU, self).__init__(
            nn.Conv2d(in_size, out_size, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU6(inplace=True),
        )


class MultisampleDropout2d(nn.Module):
    #TODO fix
    """https://arxiv.org/pdf/1905.09788.pdf"""

    def __init__(self, ratio, nb_sample):
        super(MultisampleDropout2d, self).__init__()
        self.nb_sample = nb_sample

        self.dropouts = [nn.Dropout2d(ratio) for _ in range(nb_sample)]

    def forward(self, x):
        d = [dropout(x) for dropout in self.dropouts]
        return torch.mean(torch.stack(d, dim=0), dim=0)


class MultisampleDropout1d(nn.Module):
    """https://arxiv.org/pdf/1905.09788.pdf"""

    def __init__(self, ratio, nb_sample):
        super(MultisampleDropout1d, self).__init__()
        self.nb_sample = nb_sample

        self.dropouts = [nn.Dropout(ratio) for _ in range(nb_sample)]

    def forward(self, x):
        d = [dropout(x) for dropout in self.dropouts]
        return torch.mean(torch.stack(d, dim=0), dim=0)


class MBConv(nn.Module):
    """https://arxiv.org/pdf/1905.11946.pdf"""
    def __init__(self, in_size, out_size, t, kernel_size, stride, padding):
        super(MBConv, self).__init__()
        expand_dim = in_size * t
        self.stride = stride

        self.conv = nn.Sequential(
            nn.Conv2d(in_size, expand_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(expand_dim),
            nn.ReLU6(inplace=True),

            nn.Conv2d(expand_dim, expand_dim, kernel_size=kernel_size, stride=stride, padding=padding,
                      groups=expand_dim),
            nn.BatchNorm2d(expand_dim),
            nn.ReLU6(inplace=True),

            nn.Conv2d(expand_dim, out_size, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_size),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x):
        if self.stride == 1:
            return x + self.conv(x)
        return self.conv(x)
