import numpy as np

import torch.nn.functional as F
import torch.nn as nn
import librosa

from ubs8k.datasetManager import DatasetManager, conditional_cache_v2
from DCT.layers import ConvPoolReLU, ConvReLU, ConvBNReLUPool, ConvAdvBNReLUPool, ConvBNPoolReLU6

from typing import Tuple


# ===============================================================
#    WIDE RESNET CODE
# ===============================================================
def conv_2d(ni, nf, stride=1, ks=3):
    """3x3 convolution with 1 pixel padding"""
    return nn.Conv2d(in_channels=ni, out_channels=nf, 
                     kernel_size=ks, stride=stride, 
                     padding=ks//2, bias=False)

def bn_relu_conv(ni, nf):
    """BatchNorm → ReLU → Conv2D"""
    return nn.Sequential(nn.BatchNorm2d(ni), 
                         nn.ReLU(inplace=True), 
                         conv_2d(ni, nf))

def make_group(N, ni, nf, stride):
    """Group of residual blocks"""
    start = BasicBlock(ni, nf, stride)
    rest = [BasicBlock(nf, nf) for j in range(1, N)]
    return [start] + rest


class BasicBlock(nn.Module):
    """Residual block with shortcut connection"""
    def __init__(self, ni, nf, stride=1):
        super().__init__()
        self.bn = nn.BatchNorm2d(ni)
        self.conv1 = conv_2d(ni, nf, stride)
        self.conv2 = bn_relu_conv(nf, nf)
        self.shortcut = lambda x: x
        if ni != nf:
            self.shortcut = conv_2d(ni, nf, stride, 1)
    
    def forward(self, x):
        x = F.relu(self.bn(x), inplace=True)
        r = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x) * 0.2
        return x.add_(r)
    
class WideResNet(nn.Module):
    def __init__(self, n_groups, N, n_classes, k=1, n_start=16):
        super().__init__()      
        # Increase channels to n_start using conv layer
        layers = [conv_2d(3, n_start)]
        n_channels = [n_start]
        
        # Add groups of BasicBlock(increase channels & downsample)
        for i in range(n_groups):
            n_channels.append(n_start*(2**i)*k)
            stride = 2 if i>0 else 1
            layers += make_group(N, n_channels[i], 
                                 n_channels[i+1], stride)
        
        # Pool, flatten & add linear layer for classification
        layers += [nn.BatchNorm2d(n_channels[3]), 
                   nn.ReLU(inplace=True), 
                   nn.AdaptiveAvgPool2d(1), 
                   nn.Flatten(), 
                   nn.Linear(n_channels[3], n_classes)]
        
        self.features = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.features(x)
    
def wrn_22(): 
    return WideResNet(n_groups=3, N=3, n_classes=10, k=6)

# ============================================================



class cnn0(nn.Module):
    def __init__(self, **kwargs):
        super(cnn0, self).__init__()

        self.features = nn.Sequential(
            ConvPoolReLU(1, 24, 3, 1, 1, (4, 2), (4, 2)),
            ConvPoolReLU(24, 48, 3, 1, 1, (4, 2), (4, 2)),
            ConvPoolReLU(48, 48, 3, 1, 1, (4, 2), (4, 2)),
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


class cnn03(nn.Module):
    def __init__(self,
            input_shape: Tuple[int, int] = (64, 173),
            num_classes: int = 10,
            **kwargs) -> nn.Module:
        super().__init__()

        self.features = nn.Sequential(
            ConvPoolReLU(1, 24, 3, 1, 1, (4, 2), (4, 2)),
            ConvPoolReLU(24, 48, 3, 1, 1, (4, 2), (4, 2)),
            ConvPoolReLU(48, 72, 3, 1, 1, (2, 2), (2, 2)),
            ConvPoolReLU(72, 72, 3, 1, 1, (2, 2), (2, 2)),
            ConvReLU(72, 72, 3, 1, 1),
        )

        linear_input = input_shape[0] // 64  # // 4 // 4 // 2 // 2
        linear_input *= input_shape[1] // 16
        linear_input *= 72
        print(linear_input, num_classes)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(linear_input, num_classes),
        )

    def forward(self, x):
        x = x.view(-1, 1, *x.shape[1:])

        x = self.features(x)
        x = self.classifier(x)

        return x


class cnn05(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.features = nn.Sequential(
            ConvPoolReLU(1, 24, 3, 1, 1, (4, 2), (4, 2)),
            ConvPoolReLU(24, 48, 3, 1, 1, (4, 2), (4, 2)),
            ConvPoolReLU(48, 72, 3, 1, 1, (2, 2), (2, 2)),
            ConvPoolReLU(72, 72, 3, 1, 1, (2, 2), (2, 2)),
            ConvReLU(72, 72, 3, 1, 1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(720, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = x.view(-1, 1, *x.shape[1:])

        x = self.features(x)
        x = self.classifier(x)

        return x


class cnn04(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.features = nn.Sequential(
            ConvPoolReLU(1, 24, 3, 1, 1, (4, 2), (4, 2)),
            ConvPoolReLU(24, 48, 3, 1, 1, (4, 2), (4, 2)),
            ConvReLU(48, 48, 3, 1, 1),
            ConvReLU(48, 48, 3, 1, 1),
            ConvPoolReLU(48, 72, 3, 1, 1, (2, 2), (2, 2)),
            ConvReLU(72, 72, 3, 1, 1),
            ConvReLU(72, 72, 3, 1, 1),
            ConvPoolReLU(72, 72, 3, 1, 1, (2, 2), (2, 2)),
            ConvReLU(72, 72, 3, 1, 1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(720, 10),
            #             nn.ReLU(inplace=True),
            #             nn.Dropout(0.5),
            #             nn.Linear(64, 10),
        )

    def forward(self, x):
        x = x.view(-1, 1, *x.shape[1:])

        x = self.features(x)
        x = self.classifier(x)

        return x
    
    
class cnn01(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.features = nn.Sequential(
            ConvPoolReLU(1, 64, 3, 1, 1, (4, 2), (4, 2)),
            ConvPoolReLU(64, 128, 3, 1, 1, (4, 2), (4, 2)),
            ConvPoolReLU(128, 128, 3, 1, 1, (4, 2), (4, 2)),
            ConvReLU(128, 128, 3, 1, 1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(2688, 10),
            #             nn.ReLU(inplace=True),
            #             nn.Dropout(0.5),
            #             nn.Linear(64, 10),
        )

    def forward(self, x):
        x = x.view(-1, 1, *x.shape[1:])

        x = self.features(x)
        x = self.classifier(x)

        return x

    
class cnn02(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.features = nn.Sequential(
            ConvPoolReLU(1, 32, 3, 1, 1, (4, 2), (4, 2)),
            ConvPoolReLU(32, 64, 3, 1, 1, (4, 2), (4, 2)),
            ConvPoolReLU(64, 64, 3, 1, 1, (4, 2), (4, 2)),
            ConvReLU(64, 64, 3, 1, 1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(1344, 512),
            nn.Dropout(0.3),
            nn.ReLU6(inplace=True),
            nn.Linear(512, 10)
            #             nn.ReLU(inplace=True),
            #             nn.Dropout(0.5),
            #             nn.Linear(64, 10),
        )

    def forward(self, x):
        x = x.view(-1, 1, *x.shape[1:])

        x = self.features(x)
        x = self.classifier(x)

        return x


class cnn06(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.features = nn.Sequential(
            ConvPoolReLU(1, 24, 3, 1, 1, (4, 2), (4, 2)),
            ConvPoolReLU(24, 48, 3, 1, 1, (4, 2), (4, 2)),
            ConvPoolReLU(48, 72, 3, 1, 1, (2, 2), (2, 2)),
            ConvPoolReLU(72, 72, 3, 1, 1, (2, 2), (2, 2)),
            ConvReLU(72, 72, 3, 1, 1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(720, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = x.view(-1, 1, *x.shape[1:])

        x = self.features(x)
        x = self.classifier(x)

        return x


class cnn07(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.features = nn.Sequential(
            ConvPoolReLU(1, 32, 3, 1, 1, (4, 2), (4, 2)),
            ConvPoolReLU(32, 64, 3, 1, 1, (4, 2), (4, 2)),
            ConvPoolReLU(64, 96, 3, 1, 1, (2, 2), (2, 2)),
            ConvPoolReLU(96, 96, 3, 1, 1, (2, 2), (2, 2)),
            ConvReLU(96, 96, 3, 1, 1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(960, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = x.view(-1, 1, *x.shape[1:])

        x = self.features(x)
        x = self.classifier(x)

        return x

class cnn06(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.features = nn.Sequential(
            ConvPoolReLU(1, 24, 3, 1, 1, (4, 2), (4, 2)),
            ConvPoolReLU(24, 48, 3, 1, 1, (4, 2), (4, 2)),
            ConvPoolReLU(48, 72, 3, 1, 1, (2, 2), (2, 2)),
            ConvPoolReLU(72, 72, 3, 1, 1, (2, 2), (2, 2)),
            ConvReLU(72, 72, 3, 1, 1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(720, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = x.view(-1, 1, *x.shape[1:])

        x = self.features(x)
        x = self.classifier(x)

        return x

class cnn1(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.features = nn.Sequential(
            ConvBNPoolReLU6(1, 64, 3, 1, 1, (2, 2), (2, 2), dropout=0.0),
            ConvBNPoolReLU6(64, 64, 3, 1, 1, (2, 2), (2, 2), dropout=0.3),
            ConvBNPoolReLU6(64, 64, 3, 1, 1, (2, 2), (2, 2), dropout=0.3),
            ConvBNPoolReLU6(64, 64, 3, 1, 1, (2, 2), (2, 2), dropout=0.3),
            ConvReLU(64, 64, 3, 1, 1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(2560, 10),
        )

    def forward(self, x):
        x = x.view(-1, 1, *x.shape[1:])

        x = self.features(x)
        x = self.classifier(x)

        return x
    

class cnn2(nn.Module):
    def __init__(self, **kwargs):
        super(cnn2, self).__init__()

        self.features = nn.Sequential(
            ConvBNPoolReLU6(1, 32, 3, 1, 1, (2, 2), (2, 2), dropout=0.0),
            ConvBNPoolReLU6(32, 64, 3, 1, 1, (2, 2), (2, 2), dropout=0.3),
            ConvBNPoolReLU6(64, 128, 3, 1, 1, (2, 2), (2, 2), dropout=0.3),
            ConvBNPoolReLU6(128, 256, 3, 1, 1, (2, 2), (2, 2), dropout=0.3),
            ConvBNPoolReLU6(256, 256, 3, 1, 1, (2, 2), (2, 2), dropout=0.3),
            ConvReLU(256, 256, 3, 1, 1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(2560, 10),
        )

    def forward(self, x):
        x = x.view(-1, 1, *x.shape[1:])

        x = self.features(x)
        x = self.classifier(x)

        return x
    
    
class cnn3(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.features = nn.Sequential(
            ConvBNPoolReLU6(1, 16, 3, 1, 1, (2, 2), (2, 2), dropout=0.0),
            ConvBNPoolReLU6(16, 32, 3, 1, 1, (2, 2), (2, 2), dropout=0.3),
            ConvBNPoolReLU6(32, 48, 3, 1, 1, (2, 2), (2, 2), dropout=0.3),
            ConvBNPoolReLU6(48, 64, 3, 1, 1, (2, 2), (2, 2), dropout=0.3),
            ConvBNPoolReLU6(64, 80, 3, 1, 1, (2, 2), (2, 2), dropout=0.3),
            ConvReLU(80, 80, 3, 1, 1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(800, 10),
        )

    def forward(self, x):
        x = x.view(-1, 1, *x.shape[1:])

        x = self.features(x)
        x = self.classifier(x)

        return x
    
    
class cnn4(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.features = nn.Sequential(
            ConvBNPoolReLU6(1, 16, 3, 1, 1, (2, 2), (2, 2), dropout=0.0),
            ConvReLU(16, 16, 1, 1, 1),
            ConvBNPoolReLU6(16, 32, 3, 1, 1, (2, 2), (2, 2), dropout=0.3),
            ConvReLU(32, 32, 1, 1, 1),
            ConvBNPoolReLU6(32, 48, 3, 1, 1, (2, 2), (2, 2), dropout=0.3),
            ConvReLU(48, 48, 1, 1, 1),
            ConvBNPoolReLU6(48, 64, 3, 1, 1, (2, 2), (2, 2), dropout=0.3),
            ConvReLU(64, 64, 1, 1, 1),
            ConvBNPoolReLU6(64, 80, 3, 1, 1, (2, 2), (2, 2), dropout=0.3),
            ConvReLU(80, 80, 1, 1, 1),
            ConvReLU(80, 80, 3, 1, 1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(3600, 10),
        )

    def forward(self, x):
        x = x.view(-1, 1, *x.shape[1:])

        x = self.features(x)
        x = self.classifier(x)

        return x
    
    
    
class cnn5(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.features = nn.Sequential(
            ConvBNPoolReLU6(1, 32, 3, 1, 1, (2, 2), (2, 2), dropout=0.0),
            ConvReLU(32, 32, 3, 1, 1),
            ConvBNPoolReLU6(32, 64, 3, 1, 1, (2, 2), (2, 2), dropout=0.3),
            ConvReLU(64, 64, 3, 1, 1),
            ConvBNPoolReLU6(64, 96, 3, 1, 1, (2, 2), (2, 2), dropout=0.3),
            ConvReLU(96, 96, 3, 1, 1),
            ConvBNPoolReLU6(96, 128, 3, 1, 1, (2, 2), (2, 2), dropout=0.3),
            ConvReLU(128, 128, 3, 1, 1),
            ConvBNPoolReLU6(128, 160, 3, 1, 1, (2, 2), (2, 2), dropout=0.3),
            ConvReLU(160, 160, 3, 1, 1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(1600, 10),
        )

    def forward(self, x):
        x = x.view(-1, 1, *x.shape[1:])

        x = self.features(x)
        x = self.classifier(x)

        return x

    
class cnn6(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.features = nn.Sequential(
            ConvBNPoolReLU6(1, 32, 3, 1, 1, (2, 2), (2, 2), dropout=0.0),
            ConvBNPoolReLU6(32, 64, 3, 1, 1, (2, 2), (2, 2), dropout=0.3),
            ConvBNPoolReLU6(64, 96, 3, 1, 1, (2, 2), (2, 2), dropout=0.3),
            ConvBNPoolReLU6(96, 128, 3, 1, 1, (2, 2), (2, 2), dropout=0.3),
            ConvBNPoolReLU6(128, 160, 3, 1, 1, (2, 2), (2, 2), dropout=0.3),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(1600, 10),
        )

    def forward(self, x):
        x = x.view(-1, 1, *x.shape[1:])

        x = self.features(x)
        x = self.classifier(x)

        return x

    
class cnn61(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.features = nn.Sequential(
            ConvBNPoolReLU6(1, 32, 3, 1, 1, (2, 2), (2, 2), dropout=0.0),
            ConvBNPoolReLU6(32, 64, 3, 1, 1, (2, 2), (2, 2), dropout=0.3),
            ConvBNPoolReLU6(64, 96, 3, 1, 1, (2, 2), (2, 2), dropout=0.3),
            ConvBNPoolReLU6(96, 128, 3, 1, 1, (2, 2), (2, 2), dropout=0.3),
            ConvBNPoolReLU6(128, 160, 3, 1, 1, (2, 2), (2, 2), dropout=0.3),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(1600, 256),
            nn.ReLU6(inplace=True),
            nn.DropOut(0.3),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = x.view(-1, 1, *x.shape[1:])

        x = self.features(x)
        x = self.classifier(x)

        return x

    
class cnn7(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.features = nn.Sequential(
            ConvBNPoolReLU6(1, 64, 3, 1, 1, (2, 2), (2, 2), dropout=0.0),
            ConvBNPoolReLU6(64, 128, 3, 1, 1, (2, 2), (2, 2), dropout=0.3),
            ConvBNPoolReLU6(128, 196, 3, 1, 1, (2, 2), (2, 2), dropout=0.3),
            ConvBNPoolReLU6(196, 256, 3, 1, 1, (2, 2), (2, 2), dropout=0.3),
            ConvBNPoolReLU6(256, 256, 3, 1, 1, (2, 2), (2, 2), dropout=0.3),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(2560, 10),
        )

    def forward(self, x):
        x = x.view(-1, 1, *x.shape[1:])

        x = self.features(x)
        x = self.classifier(x)

        return x


































