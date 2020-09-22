from typing import Tuple
from torch.nn import Module
from torch.nn import Sequential
from DCT.util.transforms import PadUpTo
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB


def supervised() -> Tuple[Module, Module]:
    commun_transforms = Sequential(
        PadUpTo(target_length=16000, mode="constant", value=0),
        MelSpectrogram(sample_rate=16000, n_fft=2048, hop_length=512, n_mels=64),
        AmplitudeToDB(),
    )

    train_transform = commun_transforms
    val_transform = commun_transforms

    return train_transform, val_transform


def dct() -> Tuple[Module, Module]:
    raise NotImplementedError


def dct_uniloss() -> Tuple[Module, Module]:
    raise NotImplementedError


def dct_aug4adv() -> Tuple[Module, Module]:
    raise NotImplementedError
