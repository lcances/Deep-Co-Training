import numpy as np

import torch
import torch.nn as nn
import librosa

from ubs8k.datasetManager import DatasetManager, conditional_cache_v2
from DCT.layers import ConvPoolReLU, ConvReLU, ConvBNReLUPool, ConvAdvBNReLUPool
from torchvision.models import ResNet
import torchvision.models as tm

# =============================================================================
#    WIDE RES NET
# =============================================================================
def wideresnet50_2(**kwargs):
    model = ResNet(tm.resnet.Bottleneck, [3, 4, 6, 3], num_classes=10, **kwargs)
    return model

def wideresnet28_2(**kwargs):
    model = ResNet(tm.resnet.Bottleneck, [2, 2, 2, 2], num_classes=10, **kwargs)
    return model

# =============================================================================


class Pmodel(nn.Module):
    # https://arxiv.org/pdf/1610.02242.pdf page 9, table 5
    
    def __init__(self, **kwargs):
        nn.Module.__init__(self)

        self.features = nn.Sequential(
            *self.conv_block(3, 128, (3, 3), padding=1),
            *self.conv_block(128, 128, (3, 3), padding=1),
            *self.conv_block(128, 128, (3, 3), padding=1),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            nn.Dropout2d(0.5),
            
            *self.conv_block(128, 256, (3, 3), padding=1),
            *self.conv_block(256, 256, (3, 3), padding=1),
            *self.conv_block(256, 256, (3, 3), padding=1),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            nn.Dropout2d(0.5),
            
            *self.conv_block(256, 512, (3, 3), padding=0),
            *self.conv_block(512, 256, (1, 1), padding=0),
            *self.conv_block(256, 128, (1, 1), padding=0),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(128, momentum=0.999),
            nn.utils.weight_norm( nn.Linear(128, 10) )
        )
        
    def conv_block(self, in_size, out_size, kernel, padding):
        return [
            nn.BatchNorm2d(in_size, momentum=0.999),
            nn.LeakyReLU(0.1),
            nn.utils.weight_norm( nn.Conv2d(in_size, out_size, kernel, stride=1, padding=padding) ),
        ]

    def forward(self, x):
#         x = x.view(-1, 1, *x.shape[1:])

        # add gaussian noise
        x = x + torch.randn(x.size()).cuda() * 0.15
        x = torch.clamp(x, 0, 1)
        
        x = self.features(x)
#         x = x.squeeze(dim=(-2, -1))
#         x = x.permute(0, 2, 1)
        x = self.classifier(x)

        return x
    

class cifar_cnn0(nn.Module):
    # https://arxiv.org/pdf/1610.02242.pdf page 9, table 5
    
    def __init__(self, **kwargs):
        nn.Module.__init__(self)

        self.features = nn.Sequential(
            *self.conv_block(3, 64, (3, 3), padding=1),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            nn.Dropout2d(0.3),
            
            *self.conv_block(64, 64, (3, 3), padding=1),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            nn.Dropout2d(0.3),
            
            *self.conv_block(64, 64, (1, 1), padding=0),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 10)
        )
        
    def conv_block(self, in_size, out_size, kernel, padding):
        return [
            nn.BatchNorm2d(out_size, momentum=0.999),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_size, out_size, kernel, stride=1, padding=padding),
        ]

    def forward(self, x):
#         x = x.view(-1, 1, *x.shape[1:])

        # add gaussian noise
        x = x + torch.randn(x.size()).cuda() * 0.15
        x = torch.clamp(x, 0, 1)
        
        x = self.features(x)
#         x = x.squeeze(dim=(-2, -1))
#         x = x.permute(0, 2, 1)
        x = self.classifier(x)

        return x