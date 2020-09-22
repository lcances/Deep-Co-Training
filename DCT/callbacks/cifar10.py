def lr_lambda(e):
    if e < 60:
        return 1

    elif 60 <= e < 120:
        return 0.2

    elif 120 <= e < 160:
        return 0.04

    else:
        return 0.008


def supervised(**kwargs) -> list:
    return [lr_lambda]


def dct(**kwargs) -> list:
    return [lr_lambda]


def dct_uniloss() -> list:
    return [lr_lambda]
