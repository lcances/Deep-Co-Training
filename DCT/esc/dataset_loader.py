from DCT.util.utils import ZipCycle

import random, os
import numpy as np
import torch.utils.data as torch_data
from torchaudio.datasets import ESC10, ESC50

from typing import Union, Tuple
from torch.utils.data import DataLoader

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
        return _load_supervised_helper(ESC10, dataset_root, supervised_ratio, batch_size, train_folds, val_folds, **kwargs)


def load_esc50_supervised(
        dataset_root,
        supervised_ratio: float = 1.0,
        batch_size: int = 128,
        train_folds: tuple = (1, 2, 3, 4),
        val_folds: tuple = (5, ),
        **kwargs) -> Tuple[DataLoader, DataLoader]:
        return _load_supervised_helper(ESC50, dataset_root, supervised_ratio, batch_size, train_folds, val_folds, **kwargs)


def _load_supervised_helper(
        dataset_class: Union[ESC10, ESC50],
        dataset_root,
        supervised_ratio: float = 1.0,
        batch_size: int = 128,
        train_folds: tuple = (1, 2, 3, 4),
        val_folds: tuple = (5, ),
        **kwargs
) -> Tuple[DataLoader, DataLoader]:
    """
    Load the cifar10 dataset for Deep Co Training system.
    """
    dataset_path = os.path.join(dataset_root, "ESC-50-master")
    train_dataset = dataset_class(root=dataset_path, folds=train_folds, download=True)
    val_dataset = dataset_class(root=dataset_path, folds=val_folds, download=True)
    
    # Split the training dataset into a supervised and unsupervised sets
    s_idx, u_idx = _split_s_u(train_dataset, supervised_ratio, nb_class=train_dataset.nb_class)
    
    # Calc the size of the supervised and unsupervised batch
    nb_s_file = len(s_idx)
    nb_u_file = len(u_idx)
    
    ratio = nb_s_file / nb_u_file
    s_batch_size = int(np.floor(batch_size * ratio))
    u_batch_size = int(np.ceil(batch_size * (1 - ratio)))
    
    # Create the sample, the loader and zip them
    sampler_s = torch_data.SubsetRandomSampler(s_idx)
    sampler_u = torch_data.SubsetRandomSampler(u_idx)

    train_loader_s1 = torch_data.DataLoader(train_dataset, batch_size=s_batch_size, sampler=sampler_s)
    train_loader_s2 = torch_data.DataLoader(train_dataset, batch_size=s_batch_size, sampler=sampler_s)
    train_loader_u = torch_data.DataLoader(train_dataset, batch_size=u_batch_size, sampler=sampler_u)

    train_loader = ZipCycle([train_loader_s1, train_loader_s2, train_loader_u])
    val_loader = torch_data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    
    return None, train_loader, val_loader
    