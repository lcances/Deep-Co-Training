import collections
import os

os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
import numpy as np
import pandas as pd

from ubs8k.datasetManager import DatasetManager, StaticManager
from torch.utils import data
from ubs8k.signal_augmentations import SignalAugmentation
from ubs8k.spec_augmentations import SpecAugmentation

import logging


class Dataset(data.Dataset):
    def __init__(self, manager: (DatasetManager, StaticManager), train: bool = True, val:bool = False, augments=(),
                 static_augmentation_ratio = {}, cached=False):
        super().__init__()

        self.manager = manager
        self.train = train
        self.val = val
        self.augments = augments
        self.static_augmentation_ratios = static_augmentation_ratio
        self.cached = cached

        self._check_arguments()

        if len(augments) != 0 and cached:
            logging.info("Cache system deactivate due to usage of online augmentation")
            self.cached = False


        # dataset access (combine weak and synthetic)
        if self.train:
            self.x = self.manager.audio["train"]
            self.y = self.manager.meta["train"]
        elif self.val:
            self.x = self.manager.audio["val"]
            self.y = self.manager.meta["val"]

        self.filenames = list(self.x.keys())

        # alias for verbose mode
        self.tqdm_func = self.manager.tqdm_func

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
        feat = self.manager.extract_feature(raw_audio, filename=filename, cached=self.cached)
        feat = self._apply_augmentation(feat, SpecAugmentation)
        y = np.asarray(y)

        return feat, y

    def _pad_and_crop(self, raw_audio):
        LENGTH = DatasetManager.LENGTH
        SR = self.manager.sr

        if len(raw_audio) < LENGTH * SR:
            missing = (LENGTH * SR) - len(raw_audio)
            raw_audio = np.concatenate((raw_audio, [0] * missing))

        if len(raw_audio) > LENGTH * SR:
            raw_audio = raw_audio[:LENGTH * SR]

        return raw_audio

    def _apply_augmentation(self, data, augType):
        """
        Choose the proper augmentation function depending on the type. If augmentation_style is static and augType is
        SignalAugmentation, then call the static augmentation function, otherwise call the dynamic augmentation
        function.

        It is possible to mix static and dynamic augmentation, static augmentation will be concider if the object in
        augment list is a string (not a callable)

        In case of static augmentation, the ratio must be define by the set_augment_ratio function, otherwise they are
        default to 0.5.

        :param data: the data to augment
        :param augType: The type augmentation( signal, spectrogram or image)
        :return: the augmented data
        """
        np.random.shuffle(self.augments)
        for augment in self.augments:

            # static augmentation are trigger on the signal phase and are represented by string
            if isinstance(augment, str) and augType == SignalAugmentation:
                return self._apply_static_augmentation_helper(augment, data)

            else:
                if not isinstance(augment, str):
                    return self._apply_dynamic_augmentation_helper(augment, data, augType)

        return data

    def _apply_dynamic_augmentation_helper(self, augment_func, data, augType):
        if isinstance(augment_func, augType):
            return augment_func(data)

        return data

    def _apply_static_augmentation_helper(self, augment_str, data):
        number_of_flavor = self.manager.static_augmentation["train"][augment_str].shape[0]
        flavor_to_use = np.random.randint(0, number_of_flavor)

        return self.manager.static_augmentation["train"][augment_str][flavor_to_use]

    def set_static_augment_ratio(self, ratios: dict):
        self.static_augmentation_ratios = ratios


class CoTrainingDataset(data.Dataset):
    """Must be used with the CoTrainingSampler"""
    def __init__(self, manager: (DatasetManager, StaticManager), ratio: float = 0.1, train: bool = True, val: bool = False,
            unlabel_target: bool = False, static_augmentation_ratios: dict = {},
            augments: tuple = (), S_augment: bool = True, U_augment: bool = True,
            cached:bool = False):
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
        self.S_augment = S_augment
        self.U_augment = U_augment
        self.static_augmentation_ratios = static_augmentation_ratios
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
                augment=self.S_augment
            )
            X.append(X_V)
            y.append(y_V)

        # Prepare U ---------
        target_meta = None if self.unlabel_target else self.y_U
        X_U, y_U = self._generate_data(
            U_indexes,
            target_filenames=self.filenames_U,
            target_meta=target_meta,
            augment=self.U_augment
        )
        X.append(X_U)
        y.append(y_U)

        return X, y

    def _generate_data(self, indexes: list, target_filenames: list, target_meta: pd.DataFrame = None, augment: bool = False):
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
        for i, filename in enumerate(filenames):
            if augment:
                print("Signal augmentation")
                raw_audios[i] = self._apply_augmentation(raw_audios[i], SignalAugmentation, filename)

            raw_audios[i] = self._pad_and_crop(raw_audios[i])
            feat = self.manager.extract_feature(raw_audios[i], filename=filenames[i], cached=self.cached)

            if augment:
                feat = self._apply_augmentation(feat, SpecAugmentation)

            features.append(feat)

        # Convert to np array
        return np.array(features), np.array(targets)

    def _apply_augmentation(self, data, augType, filename: str = None):
        """
        Choose the proper augmentation function depending on the type. If augmentation_style is static and augType is
        SignalAugmentation, then call the static augmentation function, otherwise call the dynamic augmentation
        function.

        It is possible to mix static and dynamic augmentation, static augmentation will be concider if the object in
        augment list is a string (not a callable)

        In case of static augmentation, the ratio must be define by the set_augment_ratio function, otherwise they are
        default to 0.5.

        :param data: the data to augment
        :param augType: The type augmentation( signal, spectrogram or image)
        :param filename: The filename of the current fiel to processe (usefull for static augmentation)
        :return: the augmented data
        """
        np.random.shuffle(self.augments)
        for augment in self.augments:
            print("augment: ", augment)
            # static augmentation are trigger on the signal phase and are represented by string
            if isinstance(augment, str) and augType == SignalAugmentation:
                print("Applying static augmentation")
                return self._apply_static_augmentation_helper(augment, data, filename)

            else:
                if not isinstance(augment, str):
                    return self._apply_dynamic_augmentation_helper(augment, data, augType)

        return data

    def _apply_dynamic_augmentation_helper(self, augment_func, data, augType):
        if isinstance(augment_func, augType):
            return augment_func(data)

        return data

    def _apply_static_augmentation_helper(self, augment_str, data, filename):
        apply = np.random.random()

        if apply <= self.static_augmentation_ratios.get(augment_str, 0.5):
            print("applying static augmentation")
            number_of_flavor = self.manager.static_augmentation["train"][augment_str][filename].shape[0]
            flavor_to_use = np.random.randint(0, number_of_flavor)

            print("augment_str: ", augment_str)
            print("flavor: %s / %s" % (flavor_to_use, number_of_flavor))

            return self.manager.static_augmentation["train"][augment_str][filename][flavor_to_use]
        return data

    def set_static_augment_ratio(self, ratios: dict):
        self.static_augmentation_ratios = ratios

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

    # load the data
    # audio_root = "../dataset/audio"
    # metadata_root = "../dataset/metadata"
    audio_root = os.path.join("E:/", "Corpus", "UrbanSound8K", "audio")
    metadata_root = os.path.join("E:/", "Corpus", "UrbanSound8K", "metadata")
    static_augment_file = os.path.join("E:/", "Corpus", "UrbanSound8K", "audio", "urbansound8k_22050_augmentations.hdf5")
    augment_list = ["I_PSC1"]
    manager = StaticManager(metadata_root, audio_root, subsampling=1.0, subsampling_method="balance",
                            static_augment_file=static_augment_file, augment_list=augment_list,
                            verbose=1, train_fold=[1], val_fold=[10])

    # prepare the sampler with the specified number of supervised file
    train_dataset = CoTrainingDataset(manager, 0.1, augments=augment_list, S_augment=True, U_augment=True, train=True, val=False)
    #val_dataset = CoTrainingDataset(manager, 1.0, train=False, val=True)
    #train_sampler = CoTrainingSampler(train_dataset, 32, nb_class=10, nb_view=2, ratio=None,
    #                                  method="duplicate")  # ratio is manually set here

    test = [train_dataset[ [[0], [1]] ][0][0] for _ in range(10)]
    for t in test:
        print(np.mean(t), np.std(t))

    #train_loader = data.DataLoader(train_dataset, batch_sampler=train_sampler)
    # val_loader = data.DataLoader(val_dataset, batch_size=128)

