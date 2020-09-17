from DCT.util.utils import ZipCycle

import random, os
import numpy as np
from torch.nn import Module
from torch import Tensor
import torch.utils.data as torch_data
from torchaudio.datasets import ESC10, ESC50

from typing import Union, Tuple
from torch.utils.data import DataLoader
from DCT.util.utils import conditional_cache_v2

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
    

def load_dct(dataset_root, supervised_ratio: float = 0.1, batch_size: int = 100, **kwargs ):
    raise NotImplementedError

# =============================================================================
#        SUPERVISED DATASETS
# =============================================================================
def load_esc10_supervised(
        dataset_root,
        supervised_ratio: float = 1.0,
        batch_size: int = 128,
        train_folds: tuple = (1, 2, 3, 4),
        val_folds: tuple = (5, ),
        **kwargs) -> Tuple[DataLoader, DataLoader]:
        return _load_supervised_helper(ESC10_NoSR, dataset_root, supervised_ratio, batch_size, train_folds, val_folds, **kwargs)


def load_esc50_supervised(
        dataset_root,
        supervised_ratio: float = 1.0,
        batch_size: int = 128,
        train_folds: tuple = (1, 2, 3, 4),
        val_folds: tuple = (5, ),
        **kwargs) -> Tuple[DataLoader, DataLoader]:
        return _load_supervised_helper(ESC50_NoSR, dataset_root, supervised_ratio, batch_size, train_folds, val_folds, **kwargs)


def _load_supervised_helper(
        dataset_class: Union[ESC10, ESC50],
        dataset_root,
        supervised_ratio: float = 1.0,
        batch_size: int = 128,
        train_folds: tuple = (1, 2, 3, 4),
        val_folds: tuple = (5, ),
        transform: Module = None,
        **kwargs) -> Tuple[DataLoader, DataLoader]:
    """
    Load the cifar10 dataset for Deep Co Training system.
    """
    num_workers = kwargs.get("num_workers", 0)
    dataset_path = os.path.join(dataset_root)

    # validation subset
    val_dataset = dataset_class(root=dataset_path, folds=val_folds, download=True, transform=transform)
    val_loader = torch_data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # Training subset
    train_dataset = dataset_class(root=dataset_path, folds=train_folds, download=True, transform=transform)

    if supervised_ratio == 1.0:
        train_loader = torch_data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    else:
        s_idx, u_idx = _split_s_u(train_dataset, supervised_ratio, nb_class=train_dataset.nb_class)
    
        sampler_s = torch_data.SubsetRandomSampler(s_idx)
        train_loader = torch_data.DataLoader(train_dataset, batch_size=batch_size, sampler=sampler_s)
    
    return None, train_loader, val_loader
    