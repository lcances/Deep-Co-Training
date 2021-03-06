from multiprocessing import Process, Manager
import logging
import datetime
import random
import numpy as np
import torch
import logging
import time
from collections import Iterable, Sized
import torch.distributed as dist

# TODO write q timer decorator that deppend on the logging level


def get_train_format(framework: str = 'supervised'):
    assert framework in ['supervised', 'mean-teacher', 'dct']

    UNDERLINE_SEQ = "\033[1;4m"
    RESET_SEQ = "\033[0m"

    if framework == 'supervised':
        header_form = "{:<8.8} {:<6.6} - {:<6.6} - {:<8.8} {:<6.6} - {:<9.9} {:<12.12}| {:<9.9}- {:<6.6}"
        value_form  = "{:<8.8} {:<6} - {:<6} - {:<8.8} {:<6.4f} - {:<9.9} {:<10.4f}| {:<9.4f}- {:<6.4f}"

        header = header_form.format(
            ".               ", "Epoch", "%", "Losses:", "ce", "metrics: ", "acc", "F1 ","Time"
        )

    elif framework == 'mean-teacher':
        header_form = "{:<8.8} {:<6.6} - {:<6.6} - {:<10.8} {:<8.6} {:<8.6} {:<8.6} {:<8.6} {:<8.6} {:<8.6} | {:<10.8} {:<8.6} {:<8.6} {:<8.6} {:<8.6} {:<8.6} - {:<8.6}"
        value_form = "{:<8.8} {:<6d} - {:<6d} - {:<10.8} {:<8.4f} {:<8.4f} {:<8.4f} {:<8.4f} {:<8.4f} {:<8.4f} | {:<10.8} {:<8.4f} {:<8.4f} {:<8.4f} {:<8.4f} {:<8.4f} - {:<8.4f}"
        header = header_form.format(".               ", "Epoch",  "%", "Student:", "ce", "ccost",
                                    "acc_s", "f1_s", "acc_u", "f1_u", "Teacher:", "ce", "acc_s", "f1_s", "acc_u", "f1_u", "Time")

    elif framework == 'dct':
        header_form = "{:<8.8} {:<6.6} - {:<6.6} - {:<8.8} {:<6.6} | {:<6.6} | {:<6.6} | {:<6.6} - {:<9.9} {:<9.9} | {:<9.9}- {:<6.6}"
        value_form = "{:<8.8} {:<6} - {:<6} - {:<8.8} {:<6.4f} | {:<6.4f} | {:<6.4f} | {:<6.4f} - {:<9.9} {:<9.4f} | {:<9.4f}- {:<6.4f}"

        header = header_form.format(
            "", "Epoch", "%", "Losses:", "Lsup", "Lcot", "Ldiff", "total", "metrics: ", "acc_s1", "acc_u1", "Time"
        )

    train_form = value_form
    val_form = UNDERLINE_SEQ + value_form + RESET_SEQ

    return header, train_form, val_form
    

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def timeit_logging(func):
    def decorator(*args, **kwargs):
        start_time = time.time()
        func(*args, **kwargs)
        logging.info("%s executed in: %.3fs" %
                     (func.__name__, time.time()-start_time))

    return decorator


def conditional_cache_v2(func):
    def decorator(*args, **kwargs):
        key_list = ",".join(map(str, args))
        key = kwargs.get("key", None)
        cached = kwargs.get("cached", None)

        if cached is not None and key is not None:
            if key not in decorator.cache.keys():
                decorator.cache[key] = func(*args, **kwargs)

            return decorator.cache[key]

        return func(*args, **kwargs)

    decorator.cache = dict()

    return decorator


def track_maximum():
    def func(key, value):
        if key not in func.max:
            func.max[key] = value
        else:
            if func.max[key] < value:
                func.max[key] = value
        return func.max[key]

    func.max = dict()
    return func


def get_datetime():
    now = datetime.datetime.now()
    return str(now)[:10] + "_" + str(now)[11:-7]


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def get_model_from_name(model_name):
    import DCT.ubs8k.models as ubs8k_models
    import DCT.ubs8k.models_test as ubs8k_models_test
    import DCT.cifar10.models as cifar10_models
    import inspect

    all_members = []
    for module in [ubs8k_models, ubs8k_models_test, cifar10_models]:
        all_members += inspect.getmembers(module)

    for name, obj in all_members:
        if inspect.isclass(obj) or inspect.isfunction(obj):
            if obj.__name__ == model_name:
                logging.info("Model loaded: %s" % model_name)
                return obj

    msg = "This model does not exist: %s\n" % model_name
    msg += "Available models are: %s" % [name for name,
                                         obj in all_members if inspect.isclass(obj) or inspect.isfunction(obj)]
    raise AttributeError("This model does not exist: %s " % msg)


def reset_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class ZipCycle(Iterable, Sized):
    """
        Zip through a list of iterables and sized objects of different lengths.
        When a iterable smaller than the longest is over, this iterator is reset to the beginning.

        Example :
        r1 = range(1, 4)
        r2 = range(1, 6)
        iters = ZipCycle([r1, r2])
        for v1, v2 in iters:
            print(v1, v2)

        will print :
        1 1
        2 2
        3 3
        1 4
        2 5
    """

    def __init__(self, iterables: list):
        for iterable in iterables:
            if len(iterable) == 0:
                raise RuntimeError("An iterable is empty.")

        self._iterables = iterables
        self._len = max([len(iterable) for iterable in self._iterables])

    def __iter__(self) -> list:
        cur_iters = [iter(iterable) for iterable in self._iterables]
        cur_count = [0 for _ in self._iterables]

        for _ in range(len(self)):
            items = []

            for i, _ in enumerate(cur_iters):
                if cur_count[i] < len(self._iterables[i]):
                    item = next(cur_iters[i])
                    cur_count[i] += 1
                else:
                    cur_iters[i] = iter(self._iterables[i])
                    item = next(cur_iters[i])
                    cur_count[i] = 1
                items.append(item)

            yield items

    def __len__(self) -> int:
        return self._len


def create_bash_crossvalidation(nb_fold: int = 10):
    cross_validation = []
    end = nb_fold

    for i in range(nb_fold):
        train_folds = []
        start = i

        for i in range(nb_fold - 1):
            start = (start + 1) % nb_fold
            start = start if start != 0 else nb_fold
            train_folds.append(start)

        cross_validation.append("-t " + " ".join(map(str, train_folds)) + " -v %d" % end)
        end = (end % nb_fold) + 1
        end = end if end != 0 else nb_fold

    print(";".join(cross_validation))

