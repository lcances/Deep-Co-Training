from ubs8k.datasetManager import DatasetManager
from ubs8k.datasets import Dataset
from DCT.util.utils import ZipCycle


import os
import numpy as np
from copy import copy
import torch.utils.data as torch_data


def supervised(
    dataset_root,
    supervised_ratio: float = 1.0,
    batch_size: int = 64,

    train_folds: tuple = (1, 2, 3, 4, 5, 6, 7, 8, 9),
    val_folds: tuple = (10, ),

    verbose=1,
    **kwargs,
):
    """
    Load the UrbanSound dataset for supervised systems.
    """
    audio_root = os.path.join(dataset_root, "UrbanSound8K", "audio")
    metadata_root = os.path.join(dataset_root, "UrbanSound8K", "metadata")

    all_folds = train_folds + val_folds

    # Create the dataset manager
    manager = DatasetManager(
        metadata_root, audio_root,
        folds=all_folds,
        verbose=verbose
    )

    # validation subset
    val_dataset = Dataset(manager, folds=val_folds, cached=True)
    val_loader = torch_data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # training subset
    train_dataset = Dataset(manager, folds=train_folds, cached=True)

    if supervised_ratio == 1.0:
        train_loader = torch_data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    else:
        s_idx, u_idx = train_dataset.split_s_u(supervised_ratio)

        # Train loader only use the s_idx
        sampler_s = torch_data.SubsetRandomSampler(s_idx)
        train_loader = torch_data.DataLoader(train_dataset, batch_size=batch_size, sampler=sampler_s)

    return manager, train_loader, val_loader


def mean_teacher(
    dataset_root,
    supervised_ratio: float = 0.1,
    batch_size: int = 64,

    train_folds: tuple = (1, 2, 3, 4, 5, 6, 7, 8, 9),
    val_folds: tuple = (10, ),

    verbose=1,
    **kwargs,
):
    assert supervised_ratio <= 1.0
    
    """
    Load the UrbanSound dataset for student teacher framework.
    """
    audio_root = os.path.join(dataset_root, "UrbanSound8K", "audio")
    metadata_root = os.path.join(dataset_root, "UrbanSound8K", "metadata")

    all_folds = train_folds + val_folds

    # Create the dataset manager
    manager = DatasetManager(
        metadata_root, audio_root,
        folds=all_folds,
        verbose=verbose
    )

    # validation subset
    val_dataset = Dataset(manager, folds=val_folds, cached=True)
    val_loader = torch_data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # training subset
    train_dataset = Dataset(manager, folds=train_folds, cached=True)
    
     # Calc the size of the Supervised and Unsupervised batch
    s_idx, u_idx = train_dataset.split_s_u(supervised_ratio)
    nb_s_file = len(s_idx)
    nb_u_file = len(u_idx)

    s_batch_size = int(np.floor(batch_size * supervised_ratio))
    u_batch_size = int(np.ceil(batch_size * (1 - supervised_ratio)))

    print("s_batch_size: ", s_batch_size)
    print("u_batch_size: ", u_batch_size)

    sampler_s = torch_data.SubsetRandomSampler(s_idx)
    sampler_u = torch_data.SubsetRandomSampler(u_idx)

    train_s_loader = torch_data.DataLoader(train_dataset, batch_size=s_batch_size, sampler=sampler_s)
    train_u_loader = torch_data.DataLoader(train_dataset, batch_size=u_batch_size, sampler=sampler_u)

    train_loader = ZipCycle([train_s_loader, train_u_loader])

    return manager, train_loader, val_loader


def dct(
        dataset_root,
        supervised_ratio: float = 0.1,
        batch_size: int = 100,

        train_folds: tuple = (1, 2, 3, 4, 5, 6, 7, 8, 9),
        val_folds: tuple = (10, ),

        verbose=1, **kwargs):
    """
    Load the urbansound dataset for Deep Co Training system.
    """
    audio_root = os.path.join(dataset_root, "UrbanSound8K", "audio")
    metadata_root = os.path.join(dataset_root, "UrbanSound8K", "metadata")

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

    s_batch_size = int(np.floor(batch_size * supervised_ratio))
    u_batch_size = int(np.ceil(batch_size * (1 - supervised_ratio)))

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
