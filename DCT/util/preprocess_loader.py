import DCT.preprocessing.esc as esc
import DCT.preprocessing.cifar10 as c10
import DCT.preprocessing.ubs8k as u8
import DCT.preprocessing.speechcommand as sc

dataset_mapper = {
    "ubs8k": {
        "supervised": u8.supervised,
        "dct": u8.dct,
        "uniloss": u8.dct_uniloss,
        "aug4adv": None,
    },

    "cifar10": {
        "supervised": c10.supervised,
        "dct": c10.dct,
        "uniloss": c10.dct_uniloss,
        "aug4adv": None,
    },

    "esc10": {
        "supervised": esc.supervised,
        "dct": esc.dct,
        "uniloss": esc.dct_uniloss,
        "aug4adv": None,
    },

    "esc50": {
        "supervised": esc.supervised,
        "dct": esc.dct,
        "uniloss": esc.dct_uniloss,
        "aug4adv": None,
    },

    "SpeechCommand": {
        "supervised": sc.supervised,
        "dct": sc.dct,
        "uniloss": sc.dct_uniloss,
        "aug4adv": None,
    },

    "gtzan": {
        "supervised": None,
        "dct": None,
        "uniloss": None,
        "aug4adv": None,
    }
}


def load_preprocesser_helper(framework: str, mapper: dict, **kwargs):
    if framework not in mapper:
        raise ValueError(f"Framework {framework} doesn't exist. Available "
                         f"framework are {list(mapper.keys())}")

    else:
        if mapper[framework] is not None:
            return mapper[framework](**kwargs)
        
        raise NotImplementedError(f"Framework {framework} is not implemented yet")


def load_preprocesser(dataset_name: str, framework: str,
                      **kwargs):
    """ The list of parameters depend on the dataset and framework used.
        See DCT.optimizer.<dataset> for more detail
    """

    if dataset_name not in dataset_mapper:
        available_dataset = ", ".join(list(dataset_mapper.keys()))
        raise ValueError(
            f"dataset {dataset_name} is not available.\n "
            f"Available datasets: {available_dataset}")

    return load_preprocesser_helper(framework, dataset_mapper[dataset_name],
                                    **kwargs)
