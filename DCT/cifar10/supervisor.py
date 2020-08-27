import torch.optim as optim

def supervised(params):
    def lr_lambda(e):
        if e < 60:
            return 1
        
        elif 60 <= e < 120:
            return 0.2
        
        elif 120 <= e < 160:
            return 0.04
        
        else:
            return 0.008
    
    lr_scheduler = LambdaLR(optimizer, lr_lambda)

    optimizers = optim.SGD(
        params = params,
        lr = 0.1,
        momentum=0.9
        weight_decay=0.0005,
    )
    
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    return optimizer, lr_scheduler


def dct(params):
    return supervised(params)