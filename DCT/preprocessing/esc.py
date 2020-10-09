from torch.nn import Sequential
from torch.nn import Module
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB

from typing import Tuple

commun_transforms = Sequential(
    MelSpectrogram(sample_rate=44100, n_fft=2048,
                   hop_length=512, n_mels=64),
    AmplitudeToDB(),
)


def supervised() -> Tuple[Module, Module]:
    train_transform = commun_transforms
    val_transform = commun_transforms

    return train_transform, val_transform


def dct() -> Tuple[Module, Module]:
    train_transform = commun_transforms
    val_transform = commun_transforms

    return train_transform, val_transform


def dct_uniloss() -> Tuple[Module, Module]:
    train_transform = commun_transforms
    val_transform = commun_transforms

    return train_transform, val_transform


def dct_aug4adv() -> Tuple[Module, Module]:
    raise NotImplementedError
