import copy
import random
import os
import torch
import numpy as np
from torch.nn import Module
from torch import Tensor
from DCT.util.utils import ZipCycle
import torch.utils.data as torch_data
from torchaudio.datasets import SPEECHCOMMANDS
from tqdm import trange

from typing import Tuple
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

URL = "speech_commands_v0.02"

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

# =============================================================================
# UTILITY FUNCTION
# =============================================================================


def _split_s_u(train_dataset, s_ratio: float = 1.0):
    _train_dataset = SpeechCommandsNoLoad.from_dataset(train_dataset)

    nb_class = len(target_mapper)
    dataset_size = len(_train_dataset)

    if s_ratio == 1.0:
        return list(range(dataset_size)), []

    s_idx, u_idx = [], []
    nb_s = int(np.ceil(dataset_size * s_ratio) // nb_class)
    cls_idx = [[] for _ in range(nb_class)]

    # To each file, an index is assigned, then they are split into classes
    for i in trange(dataset_size):
        y, _, _ = _train_dataset[i]
        cls_idx[y].append(i)

    # Recover only the s_ratio % first as supervised, rest is unsupervised
    for i in trange(len(cls_idx)):
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
    decorator.func = func
    return decorator


class SpeechCommands(SPEECHCOMMANDS):
    def __init__(self,
                 root: str,
                 subset: str = "train",
                 url: str = URL,
                 download: bool = False,
                 transform: Module = None) -> None:
        super().__init__(root, url, download, transform)

        assert subset in ["train", "validation", "testing"]
        self.subset = subset
        self._keep_valid_files()

    @cache_feature
    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        waveform, _, label, _, _ = super().__getitem__(index)
        return waveform, target_mapper[label]

    def save_cache_to_disk(self, name) -> None:
        path = os.path.join(self._path, f"{name}_features.cache")
        torch.save(self.__getitem__.cache, path)

    def load_cache_from_disk(self, name) -> bool:
        path = os.path.join(self._path, f"{name}_features.cache")

        if os.path.isfile(path):
            disk_cache = torch.load(path)
            self.__getitem__.cache.update(disk_cache)
            return True

        return False

    def _keep_valid_files(self):
        bn = os.path.basename

        def file_list(filename):
            path = os.path.join(self._path, filename)
            with open(path, "r") as f:
                to_keep = f.read().splitlines()
                return [bn(path) for path in to_keep]

        # Recover file list for validaiton and testing.
        validation_list = file_list("validation_list.txt")
        testing_list = file_list("testing_list.txt")

        # Create it for training
        training_list = [
            bn(path)
            for path in self._walker
            if bn(path) not in validation_list
            and bn(path) not in testing_list
        ]

        # Map the list to the corresponding subsets
        mapper = {
            "train": training_list,
            "validation": validation_list,
            "testing": testing_list,
        }

        self._walker = [f for f in self._walker if bn(
            f) in mapper[self.subset]]

    def __split_train_val(self, ratio: float = 0.2):
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


class SpeechCommandsNoLoad(SpeechCommands):
    @classmethod
    def from_dataset(cls, dataset: SPEECHCOMMANDS):
        root = dataset.root

        newone = cls(root=root)
        newone.__dict__.update(dataset.__dict__)
        return newone

    def _load_item(self, filepath: str, path: str) -> Tuple[Tensor, int, str,
                                                            str, int]:
        HASH_DIVIDER = "_nohash_"
        relpath = os.path.relpath(filepath, path)
        label, filename = os.path.split(relpath)
        speaker, _ = os.path.splitext(filename)

        speaker_id, utterance_number = speaker.split(HASH_DIVIDER)
        utterance_number = int(utterance_number)

        # remove Load audio
        # waveform, sample_rate = torchaudio.load(filepath)
        # return waveform, sample_rate, label, speaker_id, utterance_number
        return label, speaker_id, utterance_number

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        fileid = self._walker[index]

        label, speaker_id, utterance_number = self._load_item(
            fileid, self._path)

        return target_mapper[label], speaker_id, utterance_number


def load_dct(
    dataset_root,
    supervised_ratio: float = 0.1,
    batch_size: int = 100,

    train_transform: Module = None,
    val_transform: Module = None,

        **kwargs) -> Tuple[DataLoader, DataLoader]:

    loader_args = dict(
        num_workers=kwargs.get("num_workers", 0),
        pin_memory=kwargs.get("pin_memory", False),
    )
    dataset_path = os.path.join(dataset_root)

    # Validation subset
    val_dataset = SpeechCommands(
        root=dataset_path, subset="validation", transform=train_transform, download=True)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True, **loader_args)

    # Training subset
    train_dataset = SpeechCommands(
        root=dataset_path, subset="train", transform=val_transform, download=True)
    s_idx, u_idx = _split_s_u(train_dataset, supervised_ratio)

    # Calc the size of the Supervised and Unsupervised batch
    nb_s_file = len(s_idx)
    nb_u_file = len(u_idx)

    ratio = nb_s_file / nb_u_file
    s_batch_size = int(np.floor(batch_size * ratio))
    u_batch_size = int(np.ceil(batch_size * (1 - ratio)))

    sampler_s = SubsetRandomSampler(s_idx)
    sampler_u = SubsetRandomSampler(u_idx)

    train_loader_s1 = DataLoader(
        train_dataset, batch_size=s_batch_size, sampler=sampler_s, **loader_args)
    train_loader_s2 = DataLoader(
        train_dataset, batch_size=s_batch_size, sampler=sampler_s, **loader_args)
    train_loader_u = DataLoader(
        train_dataset, batch_size=u_batch_size, sampler=sampler_u, **loader_args)

    # combine the three loader into one
    train_loader = ZipCycle([train_loader_s1, train_loader_s2, train_loader_u])

    return None, train_loader, val_loader


def student_teacher(
        dataset_root,
        supervised_ratio: float = 0.1,
        batch_size: int = 128,

        train_transform: Module = None,
        val_transform: Module = None,

        **kwargs) -> Tuple[DataLoader, DataLoader]:
    """
    Load the SpeechCommand for a student teacher learning
    """
    loader_args = dict(
        num_workers=kwargs.get("num_workers", 0),
        pin_memory=kwargs.get("pin_memory", False),
    )
    dataset_path = os.path.join(dataset_root)

    # validation subset
    val_dataset = SpeechCommands(root=dataset_path, subset="validation", transform=train_transform, download=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, **loader_args)

    # Training subset
    train_dataset = SpeechCommands(root=dataset_path, subset="train", transform=val_transform, download=True)
    s_idx, u_idx = _split_s_u(train_dataset, supervised_ratio)
    nb_s_file = len(s_idx)
    nb_u_file = len(u_idx)

    ratio = nb_s_file / nb_u_file
    s_batch_size = int(np.floor(batch_size * ratio))
    u_batch_size = int(np.ceil(batch_size * (1 - ratio)))

    sampler_s = torch_data.SubsetRandomSampler(s_idx)
    sampler_u = torch_data.SubsetRandomSampler(u_idx)

    train_s_loader = torch_data.DataLoader(train_dataset, batch_size=s_batch_size, sampler=sampler_s)
    train_u_loader = torch_data.DataLoader(train_dataset, batch_size=u_batch_size, sampler=sampler_u)

    train_loader = ZipCycle([train_s_loader, train_u_loader])

    return None, train_loader, val_loader


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
    loader_args = dict(
        num_workers=kwargs.get("num_workers", 0),
        pin_memory=kwargs.get("pin_memory", False),
    )
    dataset_path = os.path.join(dataset_root)

    # validation subset
    val_dataset = SpeechCommands(
        root=dataset_path, subset="validation", transform=train_transform, download=True)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True, **loader_args)

    # Training subset
    train_dataset = SpeechCommands(
        root=dataset_path, subset="train", transform=val_transform, download=True)

    if supervised_ratio == 1.0:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, **loader_args)

    else:
        s_idx, u_idx = _split_s_u(train_dataset, supervised_ratio)

        sampler_s = SubsetRandomSampler(s_idx)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=sampler_s, **loader_args)

    return None, train_loader, val_loader


if __name__ == "__main__":
    _, td, vd = load_supervised("/corpus/corpus")

    print(len(td))
    print(len(vd))
