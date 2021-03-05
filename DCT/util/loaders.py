def build_mapper(modules: dict) -> dict:
    dataset_mapper = dict()

    for dataset_name, dataset_module in modules.items():
        dataset_mapper[dataset_name] = {
            "supervised": dataset_module.supervised,
            "dct": dataset_module.dct,
            "mean-teacher": dataset_module.mean_teacher,
        }

    return dataset_mapper


def load_callbacks(dataset: str, framework: str, **kwargs):
    import DCT.callbacks.esc as e
    import DCT.callbacks.ubs8k as u
    import DCT.callbacks.speechcommand as s

    # get the corresping function mapper
    dataset_mapper = build_mapper({"esc10": e, "esc50": e, "ubs8k": u, "speechcommand": s})

    return load_helper(dataset, framework, dataset_mapper, **kwargs)


def load_optimizer(dataset: str, framework: str, **kwargs):
    import DCT.optimizer.esc as e
    import DCT.optimizer.ubs8k as u
    import DCT.optimizer.speechcommand as s

    dataset_mapper = build_mapper({"esc10": e, "esc50":e, "ubs8k": u, "speechcommand": s})

    return load_helper(dataset, framework, dataset_mapper, **kwargs)


def load_preprocesser(dataset: str, framework: str, **kwargs):
    import DCT.preprocessing.esc as e
    import DCT.preprocessing.ubs8k as u
    import DCT.preprocessing.speechcommand as s

    dataset_mapper = build_mapper({"esc10": e, "esc50":e, "ubs8k": u, "speechcommand": s})

    return load_helper(dataset, framework, dataset_mapper, **kwargs)


def load_dataset(dataset: str, framework: str, **kwargs):
    import DCT.dataset_loader.esc10 as e10
    import DCT.dataset_loader.esc50 as e50
    import DCT.dataset_loader.ubs8k as u
    import DCT.dataset_loader.speechcommand as s
    
    # Default dataset for audioset is the unsupervised version
    if dataset == "audioset":
        dataset = "audioset-unbalanced"

    dataset_mapper = build_mapper(
        {"esc10": e10, "esc50":e50, "ubs8k": u, "speechcommand": s})

    return load_helper(dataset, framework, dataset_mapper, **kwargs)


def load_helper(dataset: str, framework: str, mapper: dict, **kwargs):
    _dataset = dataset.lower()
    _framework = framework.lower()

    if _dataset not in mapper.keys():
        available_dataset = "{" + " | ".join(list(mapper.keys())) + "}"
        raise ValueError(f"dataset {_dataset} is not available. Available dataset are: {available_dataset}")

    if _framework not in mapper[_dataset].keys():
        available_framework = "{" + " | ".join(list(mapper[_dataset].keys()))
        raise ValueError(f"framework {_framework} is not available. Available framework are: {available_framework}")

    return mapper[_dataset][_framework](**kwargs)
