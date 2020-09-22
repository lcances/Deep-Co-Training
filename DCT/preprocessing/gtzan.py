from torch.nn import Sequential
from torch.nn import Module
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB

from typing import Tuple


def supervised() -> Tuple[Module, Module]:
    raise NotImplementedError


def dct() -> Tuple[Module, Module]:
    raise NotImplementedError


def dct_uniloss() -> Tuple[Module, Module]:
    raise NotImplementedError


def dct_aug4adv() -> Tuple[Module, Module]:
    raise NotImplementedError
