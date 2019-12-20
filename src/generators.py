import os

os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
import numpy as np
import pandas as pd

from torch.utils import data
from datasetManager import DatasetManager


class Generator(data.Dataset):
    def __init__(self, dataset, augments=()):
        """
        Args:
            dataset:
            augments:
        """
        super().__init__()

        self.dataset = dataset
        self.augments = augments

        # dataset access (combine weak and synthetic)
        self.x = self.dataset.audio["train"]
        self.y = self.dataset.meta["train"]

        self.filenames = list(self.x.keys())

        # Validation
        self.X_val, self.y_val = None, None

        # alias for verbose mode
        self.tqdm_func = self.dataset.tqdm_func

    @property
    def validation(self):
        if self.X_val is not None and self.y_val is not None:
            return self.X_val, self.y_val

        # Need to ensure that the data are store in the same order.
        self.X_val, self.y_val = [], []
        filenames = self.dataset.audio["val"].keys()

        for filename in self.tqdm_func(filenames):
            raw_audio = self.dataset.audio["val"][filename]
            feature = self.dataset.extract_feature(raw_audio, self.dataset.sr)
            target = self.dataset.meta["val"].at[filename, "classID"]

            self.X_val.append(feature)
            self.y_val.append(target)

        self.X_val = np.asarray(self.X_val)
        self.y_val = np.asarray(self.y_val)
        return self.X_val, self.y_val

    def __len__(self):
        nb_file = len(self.filenames)
        return nb_file

    def __getitem__(self, index):
        """
        Args:
            index:
        """
        filename = self.filenames[index]
        return self._generate_data(filename)

    def _generate_data(self, filename: str):
        """
        Args:
            filename (str):
        """
        LENGTH = DatasetManager.LENGTH
        SR = self.dataset.sr

        # load the raw_audio
        raw_audio = self.x[filename]

        # recover ground truth
        y = self.y.at[filename, "classID"]

        # data augmentation
        np.random.shuffle(self.augments)
        for augment_func in self.augments:
            raw_audio = augment_func(raw_audio)

        # padding and cropping
        if len(raw_audio) < LENGTH * SR:
            missing = (LENGTH * SR) - len(raw_audio)
            raw_audio = np.concatenate((raw_audio, [0] * missing))

        if len(raw_audio) > LENGTH * SR:
            raw_audio = raw_audio[:LENGTH * SR]

        # extract feature
        feat = self.dataset.extract_feature(raw_audio, SR)
        y = np.asarray(y)

        return feat, y


class CoTrainingGenerator(data.Dataset):
    """Must be used with the CoTrainingSampler"""

    def __init__(self, dataset, sampler, unlabel_target: bool = False, augments: tuple = ()):
        """
        Args:
            dataset:
            sampler:
            unlabel_target (bool): If the unlabel target should be return or not
            augments (list):
        """
        super(CoTrainingGenerator, self).__init__()

        self.dataset = dataset
        self.sampler = sampler
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

    def _prepare_cotraining_metadata(self):
        """Using the sampler nb of of supervised file, select balanced amount of
        file in each class
        """
        metadata = self.dataset.meta["train"]
        nb_S = len(self.sampler.S_idx)

        # Prepare ground truth
        for i in range(DatasetManager.NB_CLASS):
            class_samples = metadata.loc[metadata.classID == i]

            nb_sample_S = nb_S // DatasetManager.NB_CLASS

            if i == 0:
                self.y_S = class_samples[:nb_sample_S]
                self.y_U = class_samples[nb_sample_S:]
            else:
                class_meta_S = class_samples[:nb_sample_S]
                class_meta_U = class_samples[nb_sample_S:]
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
            feature = self.dataset.extract_feature(raw_audio, self.dataset.sr)
            target = self.dataset.meta["val"].at[filename, "classID"]

            self.X_val.append(feature)
            self.y_val.append(target)

        self.X_val = np.asarray(self.X_val)
        self.y_val = np.asarray(self.y_val)

        return self.X_val, self.y_val

    def __len__(self) -> int:
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
        features = [self.dataset.extract_feature(raw, SR) for raw in raw_audios]

        # Convert to np array
        return np.array(features), np.array(targets)


class CoTrainingGenerator_SA(CoTrainingGenerator):
    def __init__(self, dataset, sampler, unlabel_target: bool = False, augments: tuple = ()):
        super().__init__(dataset, sampler, unlabel_target, augments)

    def __getitem__(self, batch_idx):
        raw_X, y = super().__getitem__(batch_idx)

        views_indexes = batch_idx[:-1]
        U_indexes = batch_idx[-1]

        # Prepare views
        augmentations = dict()
        for augment in self.augments:
            X = []

            # prepare for each view
            for vi in views_indexes:
                X_V, y_V = self._generate_data(
                    vi,
                    target_filenames=self.filenames_S,
                    target_raw=self.dataset.augmentations[augment],
                    target_meta=self.y_S,
                )
                X.append(X_V)

            # Prepare U
            target_meta = None if self.unlabel_target else self.y_U
            X_U, y_U = self._generate_data(
                U_indexes,
                target_filenames=self.filenames_U,
                target_raw=self.dataset.augmentations[augment],
                target_meta=target_meta
            )
            X.append(X_U)

            augmentations[augment] = X

        return raw_X, y, augmentations

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

        # Padding and cropping
        for i in range(len(raw_audios)):
            if len(raw_audios[i]) < LENGTH * SR:
                missing = (LENGTH * SR) - len(raw_audios[i])
                raw_audios[i] = np.concatenate((raw_audios[i], [0] * missing))

            if len(raw_audios[i]) > LENGTH * SR:
                raw_audios[i] = raw_audios[i][:LENGTH * SR]

        # Extract the features
        features = [self.dataset.extract_feature(raw, SR) for raw in raw_audios]

        # Convert to np array
        return np.array(features), np.array(targets)

if __name__ == '__main__':
    from datasetManager import DatasetManager
    from samplers import CoTrainingSampler

    # load the data
    audio_root = "../dataset/audio"
    metadata_root = "../dataset/metadata"
    #dataset = DatasetManager(metadata_root, audio_root, verbose=1)

    augmentation_to_use = ("PitchShiftChoice", "Noise")
    dataset = DatasetManager(metadata_root, audio_root,
                             train_fold=[1],
                             hdf_augments_file="urbansound8k_22050_default_config_1.hdf5",
                             augments=augmentation_to_use
                             )

    # prepare the sampler with the specified number of supervised file
    nb_train_file = len(dataset.audio["train"])
    nb_s_file = nb_train_file // 10
    nb_s_file = nb_s_file - (nb_s_file % DatasetManager.NB_CLASS)  # need to be a multiple of number of class
    nb_u_file = nb_train_file - nb_s_file
    sampler = CoTrainingSampler(32, nb_s_file, nb_u_file, nb_view=2, ratio=None, method="duplicate")

    # create the generator and the loader
    generator = CoTrainingGenerator_SA(dataset, sampler, augments=augmentation_to_use)

    train_loader = data.DataLoader(generator, batch_sampler=sampler)

    for X, y, augmented in train_loader:
        X = [x.squeeze() for x in X]
        y = [y_.squeeze() for y_ in y]
        for key in augmented:
            augmented[key] = [x.squeeze() for x in augmented[key]]

        # separate Supervised (S) and Unsupervised (U) parts
        X_S, X_U = X[:-1], X[-1]
        y_S, y_U = y[:-1], y[-1]

        print("X_U shape: ", X_U.shape)
        print("X_S shape: ", X_S[0].shape)
        print("y_U shape: ", y_U.shape)
        print("y_S shape: ", y_S[0].shape)
        break