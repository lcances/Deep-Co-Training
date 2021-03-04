from typing import Union, Tuple
from DCT.dataset.esc import ESC50
from DCT.dataset_loader.esc10 import get_supervised, get_mean_teacher, get_dct
from torch.utils.data import DataLoader


def supervised(*args, **kwargs) -> Union[None, DataLoader, DataLoader]:
    return get_supervised(ESC50)

def dct(*args, **kwargs) -> Union[None, DataLoader, DataLoader]:
    return get_dct(ESC50)

def mean_teacher(*args, **kwargs) -> Tuple[None, DataLoader, DataLoader]:
    return get_mean_teacher(ESC50)
