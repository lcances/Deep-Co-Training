import numpy as np

import torch.nn as nn
import librosa

from ubs8k.datasetManager import DatasetManager, conditional_cache_v2
from .layers import ConvPoolReLU, ConvReLU, ConvBNReLUPool, ConvAdvBNReLUPool, ConvBNPoolReLU6


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
            nn.Linear(720, 10),
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


































