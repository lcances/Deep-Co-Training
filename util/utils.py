from multiprocessing import Process, Manager
import logging
import datetime
import random
import numpy as np
import torch
import logging
import time
from collections import Iterable, Sized

# TODO write q timer decorator that deppend on the logging level
def timeit_logging(func):
    def decorator(*args, **kwargs):
        start_time = time.time()
        func(*args, **kwargs)
        logging.info("%s executed in: %.3fs" % (func.__name__, time.time()-start_time))
        
    return decorator

def feature_cache(func):
    """
    Decorator for the feature extract function. store in the memory the feature calculated, return them if call more
    than once
    IT IS NOT PROCESS / THREAD SAFE.
    Running it into multiple process will result on as mush independant cache than number of process
    """
    def decorator(*args, **kwargs):
        if "filename" in kwargs.keys() and "cached" in kwargs.keys():
            filename = kwargs["filename"]
            cached = kwargs["cached"]

            if filename is not None and cached:
                if filename not in decorator.cache.keys():
                    decorator.cache[filename] = func(*args, **kwargs)
                    return decorator.cache[filename]

                else:
                    if decorator.cache[filename] is None:
                        decorator.cache[filename] = func(*args, **kwargs)
                        return decorator.cache[filename]
                    else:
                        return decorator.cache[filename]

        return func(*args, **kwargs)

    decorator.cache = dict()

    return decorator


def multiprocess_feature_cache(func):
    """
    Decorator for the feature extraction function. Perform extraction is not already safe in memory then save it in
    memory. when call again, return feature store in memory
    THIS ONE IS PROCESS / THREAD SAFE
    """
    def decorator(*args, **kwargs):
        if "filename" in kwargs.keys() and "cached" in kwargs.keys():
            filename = kwargs["filename"]
            cached = kwargs["cached"]

            if filename is not None and cached:
                if filename not in decorator.cache.keys():
                    decorator.cache[filename] = func(*args, **kwargs)
                    return decorator.cache[filename]

                else:
                    if decorator.cache[filename] is None:
                        decorator.cache[filename] = func(*args, **kwargs)
                        return decorator.cache[filename]
                    else:
                        return decorator.cache[filename]

        return func(*args, **kwargs)

    decorator.manager = Manager()
    decorator.cache = decorator.manager.dict()

    return decorator


def get_datetime():
    now = datetime.datetime.now()
    return str(now)[:10] + "_" + str(now)[11:-7]


def get_model_from_name(model_name):
    import UrbanSound8k.models as ubs8k_models
    import UrbanSound8k.models_test as ubs8k_models_test
    import inspect

    all_members = []
    for module in [ubs8k_models, ubs8k_models_test]:
        all_members += inspect.getmembers(module)
    
    for name, obj in all_members:
        if inspect.isclass(obj) or inspect.isfunction(obj):
            if obj.__name__ == model_name:
                logging.info("Model loaded: %s" % model_name)
                return obj
    raise AttributeError("This model does not exist: %s " % model_name)


def reset_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark = False


    from typing import Iterable, Sized


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
