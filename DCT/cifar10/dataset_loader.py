from DCT.util.utils import ZipCycle

import random, os
import numpy as np
import torch.utils.data as torch_data
import torchvision.datasets
import torchvision.transforms as transforms
import hashlib

def _split_s_u(train_dataset, s_ratio: float = 0.08, nb_class: int = 10):
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
    

def load_dct(
        dataset_root,
        supervised_ratio: float = 0.1,
        batch_size: int = 100,
        **kwargs
):
    """
    Load the cifar10 dataset for Deep Co Training system.
    """
    # Prepare the default dataset
    transform = transforms.Compose(
        [transforms.ToTensor(),
#          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = torchvision.datasets.CIFAR10(root=os.path.join(dataset_root, "CIFAR10"), train=True, download=True, transform=transform)
    val_dataset = torchvision.datasets.CIFAR10(root=os.path.join(dataset_root, "CIFAR10"), train=False, download=True, transform=transform)
    
    # Split the training dataset into a supervised and unsupervised sets
    s_idx, u_idx = _split_s_u(train_dataset, supervised_ratio)
    
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


def load_supervised(
        dataset_root,
        supervised_ratio: float = 1.0,
        batch_size: int = 128,
        extra_train_transform: list = [],
        **kwargs
):
    """
    Load the cifar10 dataset for Deep Co Training system.
    """
    # Prepare the default dataset
    commun_transform = [
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ]
    
    train_transform = transforms.Compose(extra_train_transform + commun_transform)
    val_transform = transforms.Compose(commun_transform)

    train_dataset = torchvision.datasets.CIFAR10(root=os.path.join(dataset_root, "CIFAR10"), train=True, download=True, transform=train_transform)
    val_dataset = torchvision.datasets.CIFAR10(root=os.path.join(dataset_root, "CIFAR10"), train=False, download=True, transform=val_transform)
    
    # Split the training dataset into a supervised and unsupervised sets
    s_idx, u_idx = _split_s_u(train_dataset, supervised_ratio)
    
    sampler_s1 = torch_data.SubsetRandomSampler(s_idx)
    train_loader = torch_data.DataLoader(train_dataset, batch_size=batch_size, sampler=sampler_s1, num_workers=4, pin_memory=True, )
    val_loader = torch_data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, )
    
    return None, train_loader, val_loader