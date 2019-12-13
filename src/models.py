import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class repro(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)

        #         self.gaussian = GaussianNoise(sigma=0.15)
        self.features = torch.nn.Sequential(
            weight_norm(nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.1),
            weight_norm(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)),
            nn.BatchNorm2d(128, momentum=0.999),
            nn.LeakyReLU(negative_slope=0.1),
            weight_norm(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Dropout(0.5),

            weight_norm(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)),
            nn.BatchNorm2d(256, momentum=0.999),
            nn.LeakyReLU(negative_slope=0.1),
            weight_norm(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)),
            nn.BatchNorm2d(256, momentum=0.999),
            nn.LeakyReLU(negative_slope=0.1),
            weight_norm(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)),
            nn.BatchNorm2d(256, momentum=0.999),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Dropout(0.5),

            weight_norm(nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0)),
            nn.BatchNorm2d(512, momentum=0.999),
            nn.LeakyReLU(negative_slope=0.1),
            weight_norm(nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)),
            nn.BatchNorm2d(256, momentum=0.999),
            nn.LeakyReLU(negative_slope=0.1),
            weight_norm(nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)),
            nn.BatchNorm2d(128, momentum=0.999),
            nn.LeakyReLU(negative_slope=0.1),
            nn.AvgPool2d(6, stride=2, padding=0),
        )

        self.classifier = nn.Sequential(
            weight_norm(nn.Linear(128, 10))
        )

    def forward(self, x):
        x = x.view(-1, 3, 32, 32)
        #         x = self.gaussian(x)

        x = self.features(x)
        x = x.view(-1, 128)
        #         x = nn.functional.avg_pool2d(x, kernel_size=(6,6))
        #         x = x.view(-1, 128)
        x = self.classifier(x)

        return x
