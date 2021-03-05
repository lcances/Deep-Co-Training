import os
import random
import numpy as np
from typing import Callable, Union, Tuple
from DCT.util.utils import ZipCycle
from DCT.dataset.esc import ESC10, ESC50
import torch.utils.data as torch_data
from torch.nn import Module
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torch import Tensor


def _split_s_u(train_dataset, s_ratio: float = 1.0, nb_class: int = 10):
    if s_ratio == 1.0:
        return list(range(len(train_dataset))), []

    s_idx, u_idx = [], []
    nb_s = int(np.ceil(len(train_dataset) * s_ratio) // nb_class)
    cls_idx = [[] for _ in range(nb_class)]

    # To each file, an index is assigned, then they are split into classes
    for i in range(len(train_dataset)):
        _, y = train_dataset[i]
        cls_idx[y].append(i)

    for i in range(len(cls_idx)):
        random.shuffle(cls_idx[i])
        s_idx += cls_idx[i][:nb_s]
        u_idx += cls_idx[i][nb_s:]

    return s_idx, u_idx


def cache_feature(func):
    def decorator(*args, **kwargs):
        key = ",".join(map(str, args))

        if key not in decorator.cache:
            decorator.cache[key] = func(*args, **kwargs)

        return decorator.cache[key]

    decorator.cache = dict()
    return decorator


class ESC10_NoSR(ESC10):
    @cache_feature
    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        x, sr, y = super().__getitem__(index)
        return x, y


class ESC50_NoSR(ESC50):
    @cache_feature
    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        x, sr, y = super().__getitem__(index)
        return x, y


# =============================================================================
#       DEEP CO-TRAINING
# =============================================================================
def get_dct(cls: Union[ESC10, ESC50]) -> Callable:
    def dct(
            dataset_root,
            supervised_ratio: float = 0.1,
            batch_size: int = 100,
            train_folds: tuple = (1, 2, 3, 4),
            val_folds: tuple = (5, ),
            train_transform: Module = None,
            val_transform: Module = None,
            **kwargs) -> Tuple[None, DataLoader, DataLoader]:

        # Recover extra commun arguments
        num_workers = kwargs.get("num_workers", 0)
        pin_memory = kwargs.get("pin_memory", False)
        loader_args = dict(
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        
        dataset_path = os.path.join(dataset_root)

        # validation subset
        print(val_folds)
        val_dataset = cls(root=dataset_path, folds=val_folds, download=True, transform=val_transform)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, **loader_args)

        # training subset
        train_dataset = cls(root=dataset_path, folds=train_folds, download=True, transform=train_transform)

        s_idx, u_idx = _split_s_u(train_dataset, supervised_ratio, nb_class=train_dataset.nb_class)

        # Calc the size of the Supervised and Unsupervised batch
        s_batch_size = int(np.floor(batch_size * supervised_ratio))
        u_batch_size = int(np.ceil(batch_size * (1 - supervised_ratio)))

        sampler_s = SubsetRandomSampler(s_idx)
        sampler_u = SubsetRandomSampler(u_idx)

        train_loader_s1 = DataLoader(train_dataset, batch_size=s_batch_size, sampler=sampler_s, **loader_args)
        train_loader_s2 = DataLoader(train_dataset, batch_size=s_batch_size, sampler=sampler_s, **loader_args)
        train_loader_u = DataLoader(train_dataset, batch_size=u_batch_size, sampler=sampler_u, **loader_args)

        # combine the three loader into one
        train_loader = ZipCycle([train_loader_s1, train_loader_s2, train_loader_u])

        return None, train_loader, val_loader

    return dct


def dct(*args, **kwargs) -> Union[None, DataLoader, DataLoader]:
    return get_dct(ESC10)(*args, **kwargs)


# =============================================================================
#        SUPERVISED DATASETS
# ===+=========================================================================
def get_supervised(cls: Union[ESC10, ESC50]) -> Callable:
    def supervised(
            dataset_root,

            supervised_ratio: float = 1.0,
            batch_size: int = 128,
            train_folds: tuple = (1, 2, 3, 4),
            val_folds: tuple = (5, ),

            train_transform: Module = None,
            val_transform: Module = None,
            **kwargs) -> Tuple[None, DataLoader, DataLoader]:
        """
        Load the cifar10 dataset for Deep Co Training system.
        """
        # Recover extra commun arguments
        num_workers = kwargs.get("num_workers", 0)
        pin_memory = kwargs.get("pin_memory", False)
        loader_args = dict(
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        dataset_path = os.path.join(dataset_root)

        # validation subset
        val_dataset = cls(root=dataset_path, folds=val_folds, download=True, transform=val_transform)
        val_loader = torch_data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, **loader_args)

        # Training subset
        train_dataset = cls(root=dataset_path, folds=train_folds, download=True, transform=train_transform)

        if supervised_ratio == 1.0:
            train_loader = torch_data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **loader_args)

        else:
            s_idx, u_idx = _split_s_u(train_dataset, supervised_ratio, nb_class=train_dataset.nb_class)

            sampler_s = torch_data.SubsetRandomSampler(s_idx)
            train_loader = torch_data.DataLoader(train_dataset, batch_size=batch_size, sampler=sampler_s, **loader_args)

        return None, train_loader, val_loader

    return supervised


def supervised(*args, **kwargs) -> Union[None, DataLoader, DataLoader]:
    return get_supervised(ESC10)(*args, **kwargs)


# =============================================================================
#        SUPERVISED DATASETS
# =============================================================================
def get_mean_teacher(cls: Union[ESC10, ESC50]) -> callable:
    def mean_teacher(
            dataset_root,
            supervised_ratio: float = 0.1,
            batch_size: int = 128,
            train_folds: tuple = (1, 2, 3, 4),
            val_folds: tuple = (5, ),

            train_transform: Module = None,
            val_transform: Module = None,

            **kwargs) -> Tuple[None, DataLoader, DataLoader]:
        """
        Load the cifar10 dataset for Deep Co Training system.
        """
        # Recover extra commun arguments
        num_workers = kwargs.get("num_workers", 0)
        pin_memory = kwargs.get("pin_memory", False)
        loader_args = dict(
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        dataset_path = os.path.join(dataset_root)

        # validation subset
        val_dataset = cls(root=dataset_path, folds=val_folds, download=True, transform=val_transform)
        val_loader = torch_data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, **loader_args)

        # Training subset
        train_dataset = cls(root=dataset_path, folds=train_folds, download=True, transform=train_transform)
        s_idx, u_idx = _split_s_u(train_dataset, supervised_ratio, nb_class=train_dataset.nb_class)

        s_batch_size = int(np.floor(batch_size * supervised_ratio))
        u_batch_size = int(np.ceil(batch_size * (1 - supervised_ratio)))

        sampler_s = torch_data.SubsetRandomSampler(s_idx)
        sampler_u = torch_data.SubsetRandomSampler(u_idx)

        train_s_loader = torch_data.DataLoader(train_dataset, batch_size=s_batch_size, sampler=sampler_s)
        train_u_loader = torch_data.DataLoader(train_dataset, batch_size=u_batch_size, sampler=sampler_u)

        train_loader = ZipCycle([train_s_loader, train_u_loader])

        return None, train_loader, val_loader

    return mean_teacher


def mean_teacher(*args, **kwargs) -> Tuple[None, DataLoader, DataLoader]:
    return get_mean_teacher(ESC10)(*args, **kwargs)
