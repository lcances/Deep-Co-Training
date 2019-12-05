import h5py
import librosa
import os
import numpy as np
import tqdm
import time
import argparse
from multiprocessing import Pool

parser = argparse.ArgumentParser()
parser.add_argument("-sr", "--sampling_rate", default=22050, type=int, help="Librosa load sampling rate")
parser.add_argument("-l", "--length", default=4, type=int, help="audio file length")
parser.add_argument("-a", "--audioroot", default="../dataset/audio", help="path to audio folds directory")
args = parser.parse_args()

SR = args.sampling_rate
LENGTH = args.length
audio_root = args.audioroot

# create the hdf_file
hdf_path = os.path.join(audio_root, "%s_%s.hdf5" % ("urbansound8k", SR))
print("creating hdf file to : %s" % hdf_path)
hdf = h5py.File(hdf_path, 'w')

def load_file(folder_path, f):
    path = os.path.join(folder_path, f)
    raw, sr = librosa.load(path, sr=SR, res_type="kaiser_fast")

    # padding
    if len(raw) < SR * LENGTH:
        missing = (SR * LENGTH) - len(raw)
        raw = np.concatenate((raw, [0] * missing))
    elif len(raw) > SR * LENGTH:
        raw = raw[:SR*LENGTH]

    return raw


# for every folder
to_remove = [".DS_Store"]
for fold in tqdm.tqdm(range(1, 11)):
    folder_path = os.path.join(audio_root, "fold%d" % (fold))
    fold_filenames = os.listdir(folder_path)
    fold_filenames = [name for name in fold_filenames if name not in to_remove]
    dataset_shape = (len(fold_filenames), SR * LENGTH)

    # for every file, extract the raw audio at SR
    raw_audios = []

    start_time = time.time()
    pool = Pool(8)
    folder_path = [folder_path] * len(fold_filenames)

    results = pool.starmap(load_file, zip(folder_path, fold_filenames))

    results = np.array(results)
    fold_filenames = np.array(fold_filenames)

    hdf_fold = hdf.create_group("fold%d" % fold)
    hdf_fold.create_dataset("data", data=results)
    hdf_fold.create_dataset("filenames", (len(fold_filenames), ), dtype=h5py.special_dtype(vlen=str))
    for i in range(len(fold_filenames)):
        hdf_fold["filenames"][i] = fold_filenames[i]

    pool.close()
    pool.join()

hdf.close()



