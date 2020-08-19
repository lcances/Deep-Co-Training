from ubs8k.datasetManager import DatasetManager
from ubs8k.datasets import Dataset
from .utils import ZipCycle


import os
import numpy as np
import torch.utils.data as torch_data


def load_ubs8k_classic(
        dataset_root,
        supervised_ratio: float = 0.1,
        batch_size: int = 100,
        train_folds: tuple = (1, 2, 3, 4, 5, 6, 7, 8, 9),
        val_folds: tuple = (10, ),
        verbose = 1,
):
    """
    Load the urbansound dataset for Deep Co Training system.
    """
    audio_root = os.path.join(dataset_root, "ubs8k", "audio")
    metadata_root = os.path.join(dataset_root, "ubs8k", "metadata")
    
    all_folds = train_folds + val_folds
 
    # Create the dataset manager
    manager = DatasetManager(
        metadata_root, audio_root,
        folds=all_folds,
        verbose=verbose
    )
    
    # prepare the default dataset
    train_dataset = Dataset(manager, folds=train_folds, cached=True)
    val_dataset = Dataset(manager, folds=val_folds, cached=True)
    
    # split the training set into a supervised and unsupervised sets
    s_idx, u_idx = train_dataset.split_s_u(supervised_ratio)

    # Calc the size of the Supervised and Unsupervised batch
    nb_s_file = len(s_idx)
    nb_u_file = len(u_idx)

    ratio = nb_s_file / nb_u_file
    s_batch_size = int(np.floor(batch_size * ratio))
    u_batch_size = int(np.ceil(batch_size * (1 - ratio)))

    # create the sampler, the loader and "zip" them
    sampler_s1 = torch_data.SubsetRandomSampler(s_idx)
    sampler_s2 = torch_data.SubsetRandomSampler(s_idx)
    sampler_u = torch_data.SubsetRandomSampler(u_idx)

    train_loader_s1 = torch_data.DataLoader(train_dataset, batch_size=s_batch_size, sampler=sampler_s1)
    train_loader_s2 = torch_data.DataLoader(train_dataset, batch_size=s_batch_size, sampler=sampler_s2)
    train_loader_u = torch_data.DataLoader(train_dataset, batch_size=u_batch_size, sampler=sampler_u)

    train_loader = ZipCycle([train_loader_s1, train_loader_s2, train_loader_u])
    val_loader = torch_data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    
    return manager, train_loader, val_loader
