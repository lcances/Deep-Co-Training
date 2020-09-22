from torchvision import transforms
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor
from typing import Tuple


def supervised() -> Tuple[Compose, Compose]:
    train_transform = Compose(ToTensor())
    val_transform = Compose(ToTensor())

    return train_transform, val_transform


def dct() -> Tuple[Compose, Compose]:
    return supervised()


def dct_uniloss() -> Tuple[Compose, Compose]:
    return supervised()


def dct_aug4adv() -> Tuple[Compose, Compose]:
    return supervised()
