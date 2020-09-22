import copy
import random
import os
import numpy as np
from torch.nn import Module
from torch import Tensor
import torch.utils.data as torch_data
from torchaudio.datasets import SPEECHCOMMANDS

from typing import Tuple
from torch.utils.data import DataLoader

target_mapper = {
    "bed": 0,
    "bird": 1,
    "cat": 2,
    "dog": 3,
    "down": 4,
    "eight": 5,
    "five": 6,
    "follow": 7,
    "forward": 8,
    "four": 9,
    "go": 10,
    "happy": 11,
    "house": 12,
    "learn": 13,
    "left": 14,
    "marvin": 15,
    "nine": 16,
    "no": 17,
    "off": 18,
    "on": 19,
    "one": 20,
    "right": 21,
    "seven": 22,
    "sheila": 23,
    "six": 24,
    "stop": 25,
    "three": 26,
    "tree": 27,
    "two": 28,
    "up": 29,
    "visual": 30,
    "wow": 31,
    "yes": 32,
    "zero": 33,
    "backward": 34
}


def _split_s_u(train_dataset, s_ratio: float = 1.0):
    nb_class = len(target_mapper)

    if s_ratio == 1.0:
        return list(range(len(train_dataset))), []

    s_idx, u_idx = [], []
    nb_s = int(np.ceil(len(train_dataset) * s_ratio) // nb_class)
    cls_idx = [[] for _ in range(nb_class)]

    # To each file, an index is assigned, then they are split into classes
    for i in range(len(train_dataset)):
        _, y = train_dataset[i]
        cls_idx[y].append(i)

    for i in range(len(cls_idx)):
        random.shuffle(cls_idx[i])
        s_idx += cls_idx[i][:nb_s]
        u_idx += cls_idx[i][nb_s:]

    return s_idx, u_idx


def cache_feature(func):
    def decorator(*args, **kwargs):
        key = ",".join(map(str, args))

        if key not in decorator.cache:
            decorator.cache[key] = func(*args, **kwargs)

        return decorator.cache[key]

    decorator.cache = dict()
    return decorator


class SpeechCommands(SPEECHCOMMANDS):
    @cache_feature
    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        waveform, _, label, _, _ = super().__getitem__(index)
        return waveform, target_mapper[label]

    def split_train_val(self, ratio: float = 0.2):
        """ To split train and validation, let try to not have same speaker in
            both training and validation. """
        def create_metadata() -> Tuple[str, str, int, int]:
            labels, speaker_ids, utterance_numbers = [], [], []
            filepaths = []

            for i, filepath in enumerate(self._walker):
                relpath = os.path.relpath(filepath, self._path)
                label, filename = os.path.split(relpath)
                speaker, _ = os.path.splitext(filename)

                speaker_id, utterance_number = speaker.split("_nohash_")

                labels.append(label)
                speaker_ids.append(speaker_id)
                utterance_numbers.append(utterance_number)
                filepaths.append(filepath)

            labels = np.asarray(labels)
            speaker_ids = np.asarray(speaker_ids)
            utterance_numbers = np.asarray(utterance_numbers)
            filepaths = np.asarray(filepaths)

            return filepaths, labels, speaker_ids, utterance_numbers

        filepaths, labels, speaker_ids, utterance_numbers = create_metadata()

        unique_speaker = np.unique(speaker_ids)
        nb_val_speakers = int(len(unique_speaker) * ratio)

        val_speakers = unique_speaker[:nb_val_speakers]
        train_speakers = unique_speaker[nb_val_speakers:]

        train_speaker_mask = sum(
            [speaker_ids == s for s in train_speakers]) >= 1
        val_speaker_mask = sum([speaker_ids == s for s in val_speakers]) >= 1

        train_dataset = copy.deepcopy(self)
        val_dataset = copy.deepcopy(self)

        train_dataset._walker = np.asarray(self._walker)[train_speaker_mask]
        val_dataset._walker = np.asarray(self._walker)[val_speaker_mask]

        return train_dataset, val_dataset


def load_dct(dataset_root, supervised_ratio: float = 0.1,
             batch_size: int = 100, **kwargs):
    raise NotImplementedError


# =============================================================================
#        SUPERVISED DATASETS
# =============================================================================
def load_supervised(
        dataset_root,
        supervised_ratio: float = 1.0,
        batch_size: int = 128,
        train_transform: Module = None,
        val_transform: Module = None,
        **kwargs) -> Tuple[DataLoader, DataLoader]:
    """
    Load the SppechCommand for a supervised training
    """
    num_workers = kwargs.get("num_workers", 0)
    dataset_path = os.path.join(dataset_root)

    # main dataset
    dataset = SpeechCommands(
        root=dataset_path, download=True)

    train_dataset, val_dataset = dataset.split_train_val()
    train_dataset.transform = train_transform
    val_dataset.transform = val_transform

    # validation subset
    val_loader = torch_data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True)

    # Training subset
    if supervised_ratio == 1.0:
        train_loader = torch_data.DataLoader(
            train_dataset, batch_size=batch_size,
            shuffle=True, num_workers=num_workers)

    else:
        s_idx, u_idx = _split_s_u(train_dataset, supervised_ratio)

        sampler_s = torch_data.SubsetRandomSampler(s_idx)
        train_loader = torch_data.DataLoader(
            train_dataset, batch_size=batch_size, sampler=sampler_s)

    return None, train_loader, val_loader


if __name__ == "__main__":
    _, td, vd = load_supervised("/corpus/corpus")

    print(len(td))
    print(len(vd))
