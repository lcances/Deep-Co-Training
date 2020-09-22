import torch.optim


def supervised(model,
               learning_rate: float = 0.001,
               **kwargs) -> torch.optim.Optimizer:

    return torch.optim.Adam(model.parameters(), lr=learning_rate)


def dct(model1, model2,
        learning_rate: float = 0.001,
        **kwargs) -> torch.optim.Optimizer:

    parameters = list(model1.parameters()) + list(model2.parameters())
    return torch.optim.Adam(parameters(), lr=learning_rate, **kwargs)


def dct_uniloss(model1, model2,
                learning_rate: float = 0.001,
                **kwargs) -> torch.optim.Optimizer:

    parameters = list(model1.parameters()) + list(model2.parameters())
    return torch.optim.Adam(parameters(), lr=learning_rate, **kwargs)
