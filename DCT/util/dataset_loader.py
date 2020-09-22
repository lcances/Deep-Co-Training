"""
Each dataset have they specificity and split into Supervised and Unsupervised
subset is different for most of them. This utility function allow to get the
train_loader and val_loader for each dataset and framework.

This abstraction layer allow the standalone script to work with any dataset.
"""

import DCT.dataset_loader.esc as esc
import DCT.dataset_loader.cifar10 as c10
import DCT.dataset_loader.ubs8k as u8
import DCT.dataset_loader.speechcommand as SC

# import DCT.GTZAN.dataset_loader as gtzan

dataset_mapper = {
    "ubs8k": {
        "supervised": u8.load_supervised,
        "dct": u8.load_dct,
        "aug4adv": u8.load_dct_aug4adv
    },

    "cifar10": {
        "supervised": c10.load_supervised,
        "dct": c10.load_dct
    },

    "esc10": {
        "supervised": esc.load_esc10_supervised,
        # "dct": esc.load_dct,
    },

    "esc50": {
        "supervised": esc.load_esc50_supervised,
    },

    #    "gtzan": {
    #        "supervised": gtzan.load_supervised,
    #        "dct": gtzan.load_dct,
    #    },
    #
    "SpeechCommand": {
        "supervised": SC.load_supervised,
        #        "dct": SC.load_dct,
    },
}


def load_datasets_helper(framework: str, mapper: dict, **kwargs):
    if framework not in mapper:
        raise ValueError("Framework %s doesn't exist. Available framewokrs are {%s}" % (
            list(mapper.keys())))

    else:
        return mapper[framework](**kwargs)


def load_dataset(dataset_name: str, framework: str, dataset_root: str,
                 supervised_ratio: float = 0.1, batch_size: int = 100,
                 **kwargs):
    parameters = dict(
        dataset_root=dataset_root,
        supervised_ratio=supervised_ratio,
        batch_size=batch_size,
    )

    if dataset_name not in dataset_mapper:
        available_dataset = ", ".join(list(dataset_mapper.keys()))
        raise ValueError("dataset %s is not available.\n Available datasets: {%s}" % (
            dataset_name, available_dataset
        ))

    return load_datasets_helper(framework, dataset_mapper[dataset_name],
                                **parameters, **kwargs)
