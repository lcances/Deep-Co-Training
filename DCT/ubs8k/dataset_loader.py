from ubs8k.datasetManager import DatasetManager
from ubs8k.datasets import Dataset
from DCT.util.utils import ZipCycle
from DCT.augmentation_list import augmentations


import os
import numpy as np
from copy import copy
import torch.utils.data as torch_data


def load_supervised(
    dataset_root,
    supervised_ratio: float = 1.0,
    batch_size: int = 64,
    
    train_folds: tuple = (1, 2, 3, 4, 5, 6, 7 ,8, 9),
    val_folds: tuple = (10, ),
    
    verbose = 1,
    **kwargs,
):
    """
    Load the UrbanSound dataset for supervised systems.
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

    
def load_dct(
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


def load_dct_aug4adv(
        dataset_root,
        supervised_ratio: float = 0.1,
        batch_size: int = 100,
    
        train_folds: tuple = (1, 2, 3, 4, 5, 6, 7, 8, 9),
        val_folds: tuple = (10, ),
    
        augment_name_m1: str = "noise_snr20",
        augment_name_m2: str = "flip_lr",
        train_augment_ratio: float = 0.5,
    
        num_workers = 4,
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
    
    # available augmentation
    augmentation_list = list(augmentations.keys())
    
    # prepare the augmentation for both training and adversarial generation
    # /!\ NOTE: the augmentation are not yet switched
    # /!\ IT will be done in the training loop for more readability of the algorithm
    train_augmentation_m1 = copy(augmentations[augment_name_m1])
    train_augmentation_m2 = copy(augmentations[augment_name_m2])
    adv_augmentation_m1 = copy(augmentations[augment_name_m1])
    adv_augmentation_m2 = copy(augmentations[augment_name_m2])
    
    print(train_augmentation_m1)
    print(adv_augmentation_m2)
    
    # set ratio correctly (<user define> for training, 1.0 for adversarial generaion)
    train_augmentation_m1.ratio = train_augment_ratio
    train_augmentation_m2.ratio = train_augment_ratio
    adv_augmentation_m1.ratio = 1.0
    adv_augmentation_m2.ratio = 1.0
    
    # Create the augmentation training dataset
    train_dataset_m1 = Dataset(manager, folds=train_folds, augments=(train_augmentation_m1, ), cached=True)
    train_dataset_m2 = Dataset(manager, folds=train_folds, augments=(train_augmentation_m2, ), cached=True)
    val_dataset = Dataset(manager, folds=val_folds, cached=True)
    
    # Create the augmentation adversarial dataset
    adv_dataset_m1 = Dataset(manager, folds=train_folds, augments=(adv_augmentation_m1, ), cached=True)
    adv_dataset_m2 = Dataset(manager, folds=train_folds, augments=(adv_augmentation_m2, ), cached=True)
    
    
    # split the training set into a supervised and unsupervised sets
    # Any training dataset can be used
    s_idx, u_idx = train_dataset_m1.split_s_u(supervised_ratio)

    # Calc the size of the Supervised and Unsupervised batch
    nb_s_file = len(s_idx)
    nb_u_file = len(u_idx)

    ratio = nb_s_file / nb_u_file
    s_batch_size = int(np.floor(batch_size * ratio))
    u_batch_size = int(np.ceil(batch_size * (1 - ratio)))

    # create the sampler for S (m1 et m2) and U
    sampler_s1 = torch_data.SubsetRandomSampler(s_idx)
    sampler_s2 = torch_data.SubsetRandomSampler(s_idx)
    sampler_u = torch_data.SubsetRandomSampler(u_idx)

    # Apply samplers to their datasets
    train_loader_s1 = torch_data.DataLoader(train_dataset_m1, batch_size=s_batch_size, sampler=sampler_s1, num_workers=num_workers)
    train_loader_s2 = torch_data.DataLoader(train_dataset_m2, batch_size=s_batch_size, sampler=sampler_s2, num_workers=num_workers)
    adv_loader_s1 = torch_data.DataLoader(adv_dataset_m1, batch_size=s_batch_size, sampler=sampler_s1, num_workers=num_workers)
    adv_loader_s2 = torch_data.DataLoader(adv_dataset_m2, batch_size=s_batch_size, sampler=sampler_s2, num_workers=num_workers)
    
    train_loader_u1 = torch_data.DataLoader(train_dataset_m1, batch_size=u_batch_size, sampler=sampler_u, num_workers=num_workers)
    train_loader_u2 = torch_data.DataLoader(train_dataset_m2, batch_size=u_batch_size, sampler=sampler_u, num_workers=num_workers)
    adv_loader_u1 = torch_data.DataLoader(adv_dataset_m1, batch_size=u_batch_size, sampler=sampler_u, num_workers=num_workers)
    adv_loader_u2 = torch_data.DataLoader(adv_dataset_m2, batch_size=u_batch_size, sampler=sampler_u, num_workers=num_workers)

    train_loader = ZipCycle([
        train_loader_s1, train_loader_s2, train_loader_u1, train_loader_u2,
        adv_loader_s1, adv_loader_s2, adv_loader_u1, adv_loader_u2
        ])
    
    val_loader = torch_data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    
    return manager, train_loader, val_loader
