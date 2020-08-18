from ubs8k.datasetManager import ubs8k_DatasetManager
from ubs8k.datasets import ubs8k_Dataset
from utils import ZipCycle

import random
import torch.utils.data as torch_data
import torchvision.datasets.CIFAR10 as CIFAR10

def split_s_u(train_dataset, s_ratio: float = 0.1):
    if s_ratio == 1.0:
        return list(range(len(train_dataset)))
    
    # add indexes
    datasets = [(idx, x, y) for x, y in train_dataset]
    
    # separate the dataset into classes
    classes = [[] for _ in range(10)]
    for i, x, y in datasets:
        classes[y].append((i, x, y))
        
    # shuffle
    for i in range(len(classes)):
        random.shuffle(classes[i])
        
    # separate
    s_classes, u_classes = [], []
    for c in range(len(classes)):
        nb_s_file = int(np.floor(len(classes[c]) * s_ratio))
        s_classes.extend(classes[c][:nb_s_file])
        u_classes.extend(classes[c][nb_s_file:])
        
    # recover supervised and unsupervised indexes
    s_idx = [idx for idx, _, _ in s_classes]
    u_idx = [idx for idx, _, _ in u_classes]
    
    return s_idx, u_idx
    

def load_cifar10_classic(
        dataset_root,
        supervised_ratio: float = 0.1,
        batch_size: int = 100,
):
    """
    Load the cifar10 dataset for Deep Co Training system.
    """
    # Prepare the default dataset
    train_dataset = datasets.CIFAR10(root=os.path.join(dataset_root, "cifar10"), train=True, download=True, transform=None)
    val_dataset = datasets.CIFAR10(root=os.path.join(dataset_root, "cifar10"), train=False, download=True, transform=None)
    
    # Split the training dataset into a supervised and unsupervised sets
    s_idx, u_idx = split_s_u(train_dataset, supervised_ratio)
    
    # Calc the size of the supervised and unsupervised batch
    nb_s_file = len(s_idx)
    nb_u_file = len(u_idx)
    
    ratio = nb_s_file / nb_u_file
    s_batch_size = int(np.floor(batch_size * ratio))
    u_batch_size = int(np.ceil(batch_size * (1 - ratio)))
    
    # Create the sample, the loader and zip them
    sampler_s1 = torch_data.SubsetRandomSampler(s_idx)
    sampler_s2 = torch_data.SubsetRandomSampler(s_idx)
    sampler_u = torch_data.SubsetRandomSampler(u_idx)

    train_loader_s1 = torch_data.DataLoader(train_dataset, batch_size=s_batch_size, sampler=sampler_s1)
    train_loader_s2 = torch_data.DataLoader(train_dataset, batch_size=s_batch_size, sampler=sampler_s2)
    train_loader_u = torch_data.DataLoader(train_dataset, batch_size=u_batch_size, sampler=sampler_u)

    train_loader = ZipCycle([train_loader_s1, train_loader_s2, train_loader_u])
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    
    return manager, train_loader, val_loader
    
    
    return manager, train_loader, val_loader
