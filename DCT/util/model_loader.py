import DCT.ubs8k.models as ubs8k_models
import DCT.ubs8k.models_test as ubs8k_models_test
import DCT.cifar10.models as cifar10_models
import DCT.esc.models as esc_models
import inspect
import logging

def get_model_from_name(model_name):

    all_members = []
    for module in [ubs8k_models, ubs8k_models_test, cifar10_models, esc_models]:
        all_members += inspect.getmembers(module)
    
    for name, obj in all_members:
        if inspect.isclass(obj) or inspect.isfunction(obj):
            if obj.__name__ == model_name:
                logging.info("Model loaded: %s" % model_name)
                return obj
            
    msg = "This model does not exist: %s\n" % model_name
    msg += "Available models are: %s" % [name for name, obj in all_members if inspect.isclass(obj) or inspect.isfunction(obj)]
    raise AttributeError("This model does not exist: %s " % msg)