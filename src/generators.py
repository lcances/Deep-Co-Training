import collections
import os

os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
import numpy as np
import pandas as pd

from datasetManager import DatasetManager
from torch.utils import data
from signal_augmentations import SignalAugmentation
from spec_augmentations import SpecAugmentation

import logging


class Dataset(data.Dataset):
    def __init__(self, manager: DatasetManager, train: bool = True, val:bool = False, augments=(), cached=False):
        super().__init__()

        self.dataset = manager
        self.train = train
        self.val = val
        self.augments = augments

        self.cached = cached

        if len(augments) != 0 and cached:
            logging.info("Cache system deactivate due to usage of online augmentation")
            self.cached = False

        self._check_arguments()

        # dataset access (combine weak and synthetic)
        if self.train:
            self.x = self.dataset.audio["train"]
            self.y = self.dataset.meta["train"]
        elif self.val:
            self.x = self.dataset.audio["val"]
            self.y = self.dataset.meta["val"]

        self.filenames = list(self.x.keys())

        # alias for verbose mode
        self.tqdm_func = self.dataset.tqdm_func

    def _check_arguments(self):
        if sum([self.train, self.val]) != 1:
            raise AssertionError("Train and val and mutually exclusive")

    def __len__(self):
        nb_file = len(self.filenames)
        return nb_file

    def __getitem__(self, index):
        filename = self.filenames[index]
        return self._generate_data(filename)

    def _generate_data(self, filename: str):
        # load the raw_audio
        raw_audio = self.x[filename]

        # recover ground truth
        y = self.y.at[filename, "classID"]

        raw_audio = self._apply_augmentation(raw_audio, SignalAugmentation)
        raw_audio = self._pad_and_crop(raw_audio)

        # extract feature and apply spec augmentation
        feat = self.dataset.extract_feature(raw_audio, filename=filename, cached=self.cached)
        feat = self._apply_augmentation(feat, SpecAugmentation)
        y = np.asarray(y)

        return feat, y

    def _pad_and_crop(self, raw_audio):
        LENGTH = DatasetManager.LENGTH
        SR = self.dataset.sr

        if len(raw_audio) < LENGTH * SR:
            missing = (LENGTH * SR) - len(raw_audio)
            raw_audio = np.concatenate((raw_audio, [0] * missing))

        if len(raw_audio) > LENGTH * SR:
            raw_audio = raw_audio[:LENGTH * SR]

        return raw_audio

    def _apply_augmentation(self, data, augType):
        np.random.shuffle(self.augments)
        for augment_func in self.augments:
            if isinstance(augment_func, augType):
                return augment_func(data)

        return data


class CoTrainingDataset(data.Dataset):
    """Must be used with the CoTrainingSampler"""
    def __init__(self, manager: DatasetManager, ratio: float = 0.1, train: bool = True, val: bool = False,
                 unlabel_target: bool = False, augments: tuple = (), cached=False):
        """
        Args:
            manager:
            sampler:
            unlabel_target (bool): If the unlabel target should be return or not
            augments (list):
        """
        super(CoTrainingDataset, self).__init__()

        self.manager = manager
        self.ratio = ratio
        self.train = train
        self.val = val
        self.unlabel_target = unlabel_target
        self.augments = augments
        self.cached = cached

        if self.train:
            self.X = self.manager.audio["train"]
            self.y = self.manager.meta["train"]
        elif self.val:
            self.X = self.manager.audio["val"]
            self.y = self.manager.meta["val"]

        self.y_S = pd.DataFrame()
        self.y_U = pd.DataFrame()

        self._prepare_cotraining_metadata()

        self.filenames_S = self.y_S.index.values
        self.filenames_U = self.y_U.index.values

        # alias for verbose mode
        self.tqdm_func = self.manager.tqdm_func

    def _prepare_cotraining_metadata(self):
        """Using the sampler nb of of supervised file, select balanced amount of
        file in each class
        """
        # Prepare ground truth, balanced class between S and U
        for i in range(DatasetManager.NB_CLASS):
            class_samples = self.y.loc[self.y.classID == i]

            nb_sample_S = int(np.ceil(len(class_samples) * self.ratio))

            if i == 0:
                self.y_S = class_samples[:nb_sample_S]
                self.y_U = class_samples[nb_sample_S:]

            else:
                class_meta_S = class_samples[:nb_sample_S]
                class_meta_U = class_samples[nb_sample_S:]
                self.y_S = pd.concat([self.y_S, class_meta_S])
                self.y_U = pd.concat([self.y_U, class_meta_U])

    def __len__(self) -> int:
        return len(self.filenames_S) + len(self.filenames_U)

    def __getitem__(self, batch_idx):
        if isinstance(batch_idx, (list, set, tuple)):
            return self._get_train(batch_idx)
        else:
            return self._get_val(batch_idx)

    def _get_val(self, idx):
        return self._generate_data(
            [idx],
            target_filenames=self.y.index.values,
            target_meta=self.y
        )

    def _get_train(self, batch_idx):
        views_indexes = batch_idx[:-1]
        U_indexes = batch_idx[-1]

        # Prepare views --------
        X, y = [], []
        for vi in views_indexes:
            X_V, y_V = self._generate_data(
                vi,
                target_filenames=self.filenames_S,
                target_meta=self.y_S,
            )
            X.append(X_V)
            y.append(y_V)

        # Prepare U ---------
        target_meta = None if self.unlabel_target else self.y_U
        X_U, y_U = self._generate_data(
            U_indexes,
            target_filenames=self.filenames_U,
            target_meta=target_meta
        )
        X.append(X_U)
        y.append(y_U)

        return X, y

    def _generate_data(self, indexes: list, target_filenames: list, target_meta: pd.DataFrame = None):
        """
        Args:
            indexes (list):
            target_filenames (list):
            target_raw (dict):
            target_meta (pd.DataFrame):
        """
        # Get the corresponding filenames
        filenames = [target_filenames[i] for i in indexes]

        # Get the ground truth
        targets = 0
        if target_meta is not None:
            targets = [target_meta.at[name, "classID"] for name in filenames]

        # Get the raw_audio
        raw_audios = [self.X[name] for name in filenames]

        features = []
        for i in range(len(raw_audios)):
            raw_audios[i] = self._apply_augmentation(raw_audios[i], SignalAugmentation)
            raw_audios[i] = self._pad_and_crop(raw_audios[i])

            feat = self.manager.extract_feature(raw_audios[i], filename=filenames[i], cached=self.cached)
            feat = self._apply_augmentation(feat, SpecAugmentation)
            features.append(feat)

        # Convert to np array
        return np.array(features), np.array(targets)

    def _apply_augmentation(self, data, augType):
        np.random.shuffle(self.augments)
        for augment_func in self.augments:
            if isinstance(augment_func, augType):
                return augment_func(data)

        return data

    def _pad_and_crop(self, raw_audio):
        LENGTH = DatasetManager.LENGTH
        SR = self.manager.sr

        if len(raw_audio) < LENGTH * SR:
            missing = (LENGTH * SR) - len(raw_audio)
            raw_audio = np.concatenate((raw_audio, [0] * missing))

        if len(raw_audio) > LENGTH * SR:
            raw_audio = raw_audio[:LENGTH * SR]

        return raw_audio


if __name__ == '__main__':
    from samplers import CoTrainingSampler

    # load the data
    audio_root = "../dataset/audio"
    metadata_root = "../dataset/metadata"
    manager = DatasetManager(metadata_root, audio_root, subsampling=1.0, subsampling_method="balance", verbose=1, train_fold=[1], val_fold=[10])

    # prepare the sampler with the specified number of supervised file
    #train_dataset = CoTrainingDataset(manager, 0.1, train=True, val=False)
    val_dataset = CoTrainingDataset(manager, 1.0, train=False, val=True)
    #train_sampler = CoTrainingSampler(train_dataset, 32, nb_class=10, nb_view=2, ratio=None,
    #                                  method="duplicate")  # ratio is manually set here

    #train_loader = data.DataLoader(train_dataset, batch_sampler=train_sampler)
    val_loader = data.DataLoader(val_dataset, batch_size=128)

    for x, y in val_loader:
        break
