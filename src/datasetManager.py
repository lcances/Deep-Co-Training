import os
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
import numpy as np

import librosa
import tqdm
import h5py
import pandas as pd


class DatasetManager:
    class_correspondance = {"Air_conditioner": 0, "car_horn": 1, "Children_laying": 2,
                            "dog_bark": 3, "drilling": 4,
                            "engine_iddling": 5, "gun_shot": 6,
                            "jackhammer": 7, "siren": 8,
                            "street_music": 9}

    class_correspondance_reverse = dict(
        zip(
            class_correspondance.values(),
            class_correspondance.keys()
        )
    )
    NB_CLASS = 10
    LENGTH = 4

    def __init__(self, metadata_root, audio_root, hdf_file: str = None,
                 hdf_augments_file: str = None, augments: tuple = (),
                 sr: int = 22050, train_fold: list = (1, 2, 3, 4, 5, 6, 7, 8, 9),
                 val_fold: list = (10, ), verbose=1):

        self.sr = sr
        self.metadata_root = metadata_root
        self.audio_root = audio_root
        self.train_fold = train_fold
        self.val_fold = val_fold
        self.hdf_file = "urbansound8k_%s.hdf5" % self.sr if hdf_file is None else hdf_file
        self.hdf_augments_file = hdf_augments_file
        self.augments = augments
        self.feat_val = None
        self.y_val = None

        # verbose mode
        self.verbose = verbose
        if self.verbose == 1: self.tqdm_func = tqdm.tqdm
        elif self.verbose == 2: self.tqdm_func = tqdm.tqdm_notebook

        # Store the dataset metadata information
        self.meta = {}
        self._load_metadata()

        # Store the raw audio
        self.audio = {
            "train": self._hdf_to_dict(os.path.join(audio_root, self.hdf_file), train_fold),
            "val": self._hdf_to_dict(os.path.join(audio_root, self.hdf_file), val_fold)
        }

        # Preparation for the gathering the static augmentations
        if self.hdf_augments_file is not None:
            self.augmentations = {}

            for key in self.augments:
                self.augmentations[key] = self._hdf_to_dict(os.path.join(audio_root, self.hdf_augments_file), train_fold, key)

    @property
    def validation(self):
        raise NotImplementedError()

    def _hdf_to_dict(self, hdf_path, folds: list, key: str = "data") -> dict:
        output = dict()

        # open hdf file
        hdf = h5py.File(hdf_path, "r")

        for fold in self.tqdm_func(folds):
            hdf_fold = hdf["fold%d" % fold]

            filenames = hdf_fold["filenames"]
            raw_audios = hdf_fold[key]

            fold_dict = dict(zip(filenames, raw_audios))
            output = dict(**output, **fold_dict)

        # close hdf file
        hdf.close()
        return output

    def load_audio(self, file_path):
        raw_data, sr = librosa.load(file_path, sr=DatasetManager.SR, res_type="kaiser_fast")
        return raw_data, sr

    def extract_feature(self, raw_data, sr):
        feat = librosa.feature.melspectrogram(
            raw_data, sr, n_fft=2048, hop_length=512, n_mels=64, fmin=0, fmax=sr // 2)
        feat = librosa.power_to_db(feat, ref=np.max)
        return feat

    def _load_metadata(self):
        metadata_path = os.path.join(self.metadata_root, "UrbanSound8K.csv")

        data = pd.read_csv(metadata_path, sep=",")
        data = data.set_index("slice_file_name")

        self.meta["train"] = data.loc[data.fold.isin(self.train_fold)]
        self.meta["val"] = data.loc[data.fold.isin(self.val_fold)]


if __name__ == '__main__':
    audio_root = "../dataset/audio"
    metadata_root = "../dataset/metadata"
    #dataset = DatasetManager(metadata_root, audio_root)

    augmentation_to_load = ("PitchShiftChoice", "Noise")
    dataset = DatasetManager(metadata_root, audio_root,
                             hdf_augments_file="urbansound8k_22050_default_config_1.hdf5",
                             augments=("PitchShiftChoice", "Noise")
                             )

