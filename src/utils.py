import datetime
import logging


def get_datetime():
    now = datetime.datetime.now()
    return str(now)[:10] + "_" + str(now)[11:-7]


def get_model_from_name(model_name):
    import models
    import inspect

    for name, obj in inspect.getmembers(models):
        if inspect.isclass(obj) or inspect.isfunction(obj):
            if obj.__name__ == model_name:
                logging.info("Model loaded: %s" % model_func.__name__)
                return obj
    raise AttributeError("This model does not exist: %s " % model_name)
