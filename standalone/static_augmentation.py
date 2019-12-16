
import json
import h5py
import librosa
import inspect
import os
import sys
import numpy as np
import tqdm
import time
import argparse
from multiprocessing import Pool

sys.path.append("../scr")
import signal_augmentations as sa

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", default="config/default_config.json", type=str)
parser.add_argument("-r", "--repeat", default=1, type=int)
args = parser.parse_args()

# Prepare for the augmentations
with open(args.config) as json_file:
    config = json.load(json_file)


def get_augment_from_name(class_name):
    for name, obj in inspect.getmembers(sa):
        if inspect.isclass(obj):
            if obj.__name__ == class_name:
                return obj
    raise AttributeError("This augmentation method doesn't exist: (%s)" % class_name)

# Get the general information
SR = config["general"]["sampling_rate"]
LENGTH = config["general"]["length"]
audio_root = config["general"]["audioroot"]
config_name = os.path.basename(args.config)

# Create the list of augmentation function
augments = []
augment_config = config["augments"]

for key in augment_config:
    augment_obj = get_augment_from_name(key)
    augment_func = augment_obj(**augment_config[key])
    augments.append(augment_func)

# create the hdf_file
hdf_path = os.path.join(audio_root, "%s_%s_%s_%s.hdf5" % (
    "urbansound8k", SR, config_name, 1))
print("creating hdf file to : %s" % hdf_path)
hdf = h5py.File(hdf_path, 'w')

def load_file(folder_path, f, augment_func = None):
    output = dict()

    path = os.path.join(folder_path, f)
    raw, sr = librosa.load(path, sr=SR, res_type="kaiser_fast")

    if augment_func is not None:
        raw = augment_func(raw)

    # padding
    if len(raw) < SR * LENGTH:
        missing = (SR * LENGTH) - len(raw)
        raw = np.concatenate((raw, [0] * missing))
    elif len(raw) > SR * LENGTH:
        raw = raw[:SR*LENGTH]

    return raw


def process_pool_results(results, fold_filenames, subset):
    results = np.array(results)
    fold_filenames = np.array(fold_filenames)

    # create group if it doesn't exist yet
    if "fold%d" % fold not in hdf:
        hdf_fold = hdf.create_group("fold%d" % fold)
    else:
        hdf_fold = hdf["fold%s" % fold]
        hdf_fold.create_dataset(subset, data=results)

    # create the list of filenames only once (when the raw data are process)
    if subset == "raw":
        hdf_fold.create_dataset("filenames", (len(fold_filenames), ), dtype=h5py.special_dtype(vlen=str))

        for i in range(len(fold_filenames)):
            hdf_fold["filenames"][i] = fold_filenames[i]


# for every folder
to_remove = [".DS_Store"]
for fold in tqdm.tqdm(range(1, 11)):
    folder_path = os.path.join(audio_root, "fold%d" % (fold))
    fold_filenames = os.listdir(folder_path)
    fold_filenames = [name for name in fold_filenames if name not in to_remove]
    dataset_shape = (len(fold_filenames), SR * LENGTH)

    # for every file, extract the raw audio at SR
    pool = Pool(8)

    # raw audio
    folder_path = [folder_path] * len(fold_filenames)
    augment_func = [None] * len(fold_filenames)
    results = pool.starmap(load_file, zip(folder_path, fold_filenames, augment_func))
    process_pool_results(results, fold_filenames, "raw")

    # augmented audio
    for i, key in enumerate(augment_config):
        augment_func = [augments[i]] * len(fold_filenames)
        results = pool.starmap(load_file, zip(folder_path, fold_filenames, augment_func))
        process_pool_results(results, fold_filenames, key)

    pool.close()
    pool.join()

hdf.close()



