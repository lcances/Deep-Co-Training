import torch
import torch.nn as nn

from layers import ConvPoolReLU, ConvReLU


class cnn(nn.Module):
    def __init__(self):
        super(cnn, self).__init__()

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


class cnn2(nn.Module):
    def __init__(self):
        super(cnn2, self).__init__()

        self.features = nn.Sequential(
            ConvPoolReLU(1, 24, 5, 1, 1, (4, 2), (4, 2)),
            ConvPoolReLU(24, 48, 5, 1, 1, (4, 2), (4, 2)),
            ConvReLU(48, 48, 5, 1, 1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(1872, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = x.view(-1, 1, *x.shape[1:])

        x = self.features(x)
        x = self.classifier(x)

        return x