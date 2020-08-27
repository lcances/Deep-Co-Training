import torch.optim
import torch.optim.lr_scheduler as lr_scheduler

def _get_cifar10_optimizer(framework: str) -> (torch.optim, lr_scheduler):
    from DCT.cifar10.train_parameters import supervised, dct

    if framework == "supervised":
        return supervised()

    elif framework == "dct":
        return dct()

    else:
        available_framework = ["supervised", "dct"]
        raise ValueError("Framework %s is not available.\n Available framework: %s" ", ".join(available_framework))

def _get_ubs8k_optimizer(framework: str) -> (torch.optim, lr_scheduler):
    from DCT.ubs8k.train_parameters import supervised, dct, dct_uniloss

    if framework == "supervised"):
        return supervised()

    elif framework == "dct":
        return dct()

    elif framework == "dct_uniloss":
        return dct_uniloss()

    else:
        available_framework = ["supervised", "dct"]
        raise ValueError("Framework %s is not available.\n Available framework: %s" ", ".join(available_framework))

def get_optimizer(dataset: str, framework: str) -> (torch.optim, lr_scheduler):
    if dataset == "cifar10":
        return _get_cifar10_optimizer(framework)
    
    elif dataset == "ubs8k":
        return _get_ubs8k_optimizer(framework)

    else:
        available_dataset = [ "ubs8k", "cifar10" ]
        raise ValueError("dataset %s is not available.\n Available datasets: %s" % ", ".join(available_dataset))"