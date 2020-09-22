import torch.optim


def supervised(model,
               learning_rate: float = 0.1,
               momentum: float = 0.9,
               weight_decay: float = 0.0005,
               **kwargs) -> torch.optim.Optimizer:
    return torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)


def dct(model1, model2,
        learning_rate: float = 0.1,
        momentum: float = 0.9,
        weight_decay: float = 0.0005,
        **kwargs) -> torch.optim.Optimizer:

    parameters = list(model1.parameters()) + list(model2.parameters())
    return torch.optim.SGD(parameters, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)


def dct_uniloss(model1, model2,
                learning_rate: float = 0.1,
                momentum: float = 0.9,
                weight_decay: float = 0.0005,
                **kwargs) -> torch.optim.Optimizer:

    parameters = list(model1.parameters()) + list(model2.parameters())
    return torch.optim.SGD(parameters, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)