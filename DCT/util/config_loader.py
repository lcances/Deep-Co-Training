"""
The parameters for the best system for every dataset and every framework is describe into configuration files.
This file are written in YAML and can be found:
    - cifar10/config
    - ubs8k/config

This add every element of the configuration file into a NameSpace the same way argparse do.
It allow the usage of both argparse and configuration files.
"""
import yaml

class NameSpace:
    def __init__(**kwargs):
        self.__dict__.update(kwargs)

def load_config(path: str):
    with open(path) as yml_file:
        config = yaml.safe_load(yml_file)

    return NameSpace(**config)