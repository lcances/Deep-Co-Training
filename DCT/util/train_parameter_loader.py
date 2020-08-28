from DCT.cifar10.train_parameters import supervised, dct

def _get_cifar10_optimizer(framework):
    if framework == "supervised":
        return supervised()

    elif framework == "dct":
        return dct

def _get_ubs8k_optimizer(framework):
    raise NotImplementedError()

def get_optimizer(dataset, framework):
    if dataset == "cifar10":
        return _get_cifar10_optimizer(framework)
    
    elif dataset == "ubs8k":
        return _get_ubs8k_optimizer(framework)

    else:
        available_dataset = [ "ubs8k", "cifar10" ]
        raise ValueError("dataset %s is not available.\n Available datasets: %s" % ", ".join(available_dataset))"