
def load_cifar10_datasets(framework: str, **kwargs):
    from DCT.cifar10.loader import load_cifar10_dct, load_cifar10_supervised
    
    if framework == "dct":
        return load_cifar10_dct(**kwargs)

    elif framework == "supervised":
        return load_cifar10_supervised(**kwargs)

    else:
        available = ["supervised", "dct"]
        raise ValueError("framework %s do not exist. Available frameworks %s" % (framework, available))

def load_ubs8k_datasets(framework: str):
    from DCT.ubs8k.loader import load_ubs8k_dct, load_ubs8k_supervised, load_ubs8k_dct_aug4adv
        
    if framework == "dct":
        return load_ubs8k_dct(**kwargs)

    elif framework == "supervised":
        return load_ubs8k_supervised(**kwargs)

    elif framework == "aug4adv":
        return load_ubs8k_dct_aug4adv(**kwargs)

    else:
        raise ValueError("framework %s do not exist. Available %s" % (framework, available))

def load_dataset(dataset_name: str, framework: str, dataset_root: str,
        supervised_ratio: float = 0.1, batch_size: int = 100,
        **kwargs
    ):
    parameters = dict(
        dataset_root = dataset_root,
        supervised_ratio = supervised_ratio,
        batch_size = batch_size,
    )
    
    if dataset_name == "ubs8k":
        return load_ubs8k_datasets(framework, **parameters, **kwargs)
    
    elif dataset_name == "cifar10":
        return load_cifar10_datasets(framework, **parameters, **kwargs)
    
    else:
        available_dataset = [ "ubs8k", "cifar10" ]
        raise ValueError("dataset %s is not available.\n Available datasets: %s" % ", ".join(available_dataset))