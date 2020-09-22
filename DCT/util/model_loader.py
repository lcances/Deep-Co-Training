import DCT.models.ubs8k as u8
import DCT.models.ubs8k_test as u8_test
import DCT.models.cifar10 as c10
import DCT.models.esc as esc
import DCT.models.speechcommands as sc
import DCT.models.gtzan as gtzan

import inspect
import logging

dataset_mapper = {
    "ubs8k": [u8, u8_test],
    "cifar10": [c10],
    "esc10": [esc],
    "esc50": [esc],
    "speechcommand": [sc],
    "gtzan": [gtzan]
}


def get_model_from_name(model_name, module_list):

    all_members = []
    for module in module_list:
        all_members += inspect.getmembers(module)

    for name, obj in all_members:
        if inspect.isclass(obj) or inspect.isfunction(obj):
            if obj.__name__ == model_name:
                logging.info("Model loaded: %s" % model_name)
                return obj

    # Error message if model doesn't exist for the dataset
    available_models = [name for name, obj in all_members if inspect.isclass(obj) or inspect.isfunction(obj)]
    msg = f"Model {model_name} doesn't exist for this dataset\n"
    msg += f"Available models are: {available_models}"
    raise ValueError(msg)


def load_model(dataset_name: str, model_name: str):
    if dataset_name not in dataset_mapper:
        available_dataset = ", ".join(list(dataset_mapper.keys()))
        raise ValueError(
            f"Dataset {dataset_name} is not available.\n Available datasets: {available_dataset}")

    return get_model_from_name(model_name, dataset_mapper[dataset_name])
