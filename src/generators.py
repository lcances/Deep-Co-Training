import os

os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
import numpy as np
import pandas as pd

from torch.utils import data
from signal_augmentations import SignalAugmentation
from spec_augmentations import SpecAugmentation


class Generator(data.Dataset):
    def __init__(self, dataset, train:bool = True, val:bool = False, sampling: float = 1.0, augments=()):
        super().__init__()

        self.dataset = dataset
        self.train = train
        self.val = val
        self.sampling = sampling
        self.augments = augments

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

        if 0 > self.sampling > 1.0:
            raise AssertionError("Sampling ratio must be between 0 and 1")

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

        raw_audio = self._apply_signal_augmentation(raw_audio)
        raw_audio = self._pad_and_crop(raw_audio)

        # extract feature and apply spec augmentation
        feat = self.dataset.extract_feature(raw_audio)
        feat = self._apply_spec_augmentation(feat)
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

    def _apply_signal_augmentation(self, raw_audio):
        np.random.shuffle(self.augments)
        for augment_func in self.augments:
            if isinstance(augment_func, SignalAugmentation):
                raw_audio = augment_func(raw_audio)

        return raw_audio

    def _apply_spec_augmentation(self, feature):
        np.random.shuffle(self.augments)
        for augment_func in self.augments:
            if isinstance(augment_func, SpecAugmentation):
                feature = augment_func(feature)

        return feature


class CoTrainingGenerator(data.Dataset):
    """Must be used with the CoTrainingSampler"""
    def __init__(self, dataset, ratio, unlabel_target: bool = False, augments: tuple = ()):
        """
        Args:
            dataset:
            sampler:
            unlabel_target (bool): If the unlabel target should be return or not
            augments (list):
        """
        super(CoTrainingGenerator, self).__init__()

        self.dataset = dataset
        self.ratio = ratio
        self.unlabel_target = unlabel_target
        self.augments = augments

        # prepare co-training variable
        self.X_S = {}
        self.X_U = {}

        self.y_S = pd.DataFrame()
        self.y_U = pd.DataFrame()

        self._prepare_cotraining_metadata()

        self.filenames_S = self.y_S.index.values
        self.filenames_U = self.y_U.index.values
        
        # Validation
        self.X_val, self.y_val = None, None
        
        # alias for verbose mode
        self.tqdm_func = self.dataset.tqdm_func
        

        # Validation
        self.X_val, self.y_val = None, None

        # alias for verbose mode
        self.tqdm_func = self.dataset.tqdm_func

    def _prepare_cotraining_metadata(self):
        """Using the sampler nb of of supervised file, select balanced amount of
        file in each class
        """
        metadata = self.dataset.meta["train"]

        # Prepare ground truth, balanced class between S and U
        for i in range(DatasetManager.NB_CLASS):
            class_samples = metadata.loc[metadata.classID == i]

            nb_sample_S = int(np.ceil(len(class_samples) * self.ratio))
            print("generator, nb_sample_s: ", nb_sample_S)

            if i == 0:
                self.y_S = class_samples[:nb_sample_S]
                self.y_U = class_samples[nb_sample_S:]

            else:
                class_meta_S = class_samples[:nb_sample_S]
                class_meta_U = class_samples[nb_sample_S:]
                print(len(class_samples))
                print(len(class_meta_S), len(class_meta_U))
                self.y_S = pd.concat([self.y_S, class_meta_S])
                self.y_U = pd.concat([self.y_U, class_meta_U])

    @property
    def validation(self):
        if self.X_val is not None and self.y_val is not None:
            return self.X_val, self.y_val

        # Need to ensure that the data are store in the same order.
        self.X_val, self.y_val = [], []
        filenames = self.dataset.audio["val"].keys()

        for filename in self.tqdm_func(filenames):
            raw_audio = self.dataset.audio["val"][filename]
            feature = self.dataset.extract_feature(raw_audio)
            target = self.dataset.meta["val"].at[filename, "classID"]

            self.X_val.append(feature)
            self.y_val.append(target)

        self.X_val = np.asarray(self.X_val)
        self.y_val = np.asarray(self.y_val)

        return self.X_val, self.y_val

    def invalid(self):
        self.X_val = None
        self.y_val = None

    def __len__(self) -> int:
        pass
        return len(self.sampler)

    def __getitem__(self, batch_idx):
        """
        Args:
            batch_idx:
        """
        views_indexes = batch_idx[:-1]
        U_indexes = batch_idx[-1]

        # Prepare views
        X, y = [], []
        for vi in views_indexes:
            X_V, y_V = self._generate_data(
                vi,
                target_filenames=self.filenames_S,
                target_raw=self.dataset.audio["train"],
                target_meta=self.y_S,
            )
            X.append(X_V)
            y.append(y_V)

        # Prepare U
        target_meta = None if self.unlabel_target else self.y_U
        X_U, y_U = self._generate_data(
            U_indexes,
            target_filenames=self.filenames_U,
            target_raw=self.dataset.audio["train"],
            target_meta=target_meta
        )
        X.append(X_U)
        y.append(y_U)

        return X, y

    def _generate_data(self, indexes: list, target_filenames: list, target_raw: dict, target_meta: pd.DataFrame = None):
        """
        Args:
            indexes (list):
            target_filenames (list):
            target_raw (dict):
            target_meta (pd.DataFrame):
        """
        LENGTH = DatasetManager.LENGTH
        SR = self.dataset.sr
        # Get the corresponding filenames
        filenames = [target_filenames[i] for i in indexes]

        # Get the raw_adio
        raw_audios = [target_raw[name] for name in filenames]

        # Get the ground truth
        # For unlabel set, create a fake ground truth (batch collate error otherwise)
        targets = 0 #[[0] * DatasetManager.NB_CLASS for _ in range(len(filenames))]
        if target_meta is not None:
            targets = [target_meta.at[name, "classID"] for name in filenames]

        # Data augmentation ?
        for i in range(len(raw_audios)):
            np.random.shuffle(self.augments)
            for augment_func in self.augments:
                raw_audios[i] = augment_func(raw_audios[i])

        # Padding and cropping
        for i in range(len(raw_audios)):
            if len(raw_audios[i]) < LENGTH * SR:
                missing = (LENGTH * SR) - len(raw_audios[i])
                raw_audios[i] = np.concatenate((raw_audios[i], [0] * missing))

            if len(raw_audios[i]) > LENGTH * SR:
                raw_audios[i] = raw_audios[i][:LENGTH * SR]

        # Extract the features
        features = [self.dataset.extract_feature(raw) for raw in raw_audios]

        # Convert to np array
        return np.array(features), np.array(targets)

