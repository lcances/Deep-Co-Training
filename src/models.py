import torch
import torch.nn as nn

from layers import ConvPoolReLU, ConvReLU


class cnn(nn.Module):
    """https: // arxiv.org / pdf / 1608.04363.pdf"""
    def __init__(self):
        super(cnn, self).__init__()

        self.features = nn.Sequential(
            ConvPoolReLU(1, 24, 3, 1, 1),
            ConvPoolReLU(24, 48, 3, 1, 1),
            ConvPoolReLU(48, 48, 3, 1, 1),
            ConvReLU(48, 48, 3, 1, 1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(1008, 10),
        )

    def forward(self, x):
        x = x.view(-1, 1, *x.shape[1:])

        x = self.features(x)
        x = self.classifier(x)

        return x
