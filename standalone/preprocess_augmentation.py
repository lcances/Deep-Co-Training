
import h5py
import librosa
import os, sys
import numpy as np
import tqdm
import time
import argparse
from multiprocessing import Pool
from ubs8k.augmentation_list import reverse_unique_augment

import ubs8k.signal_augmentations as signal_augmentations

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--audio_root", default="../dataset/audio", help="path to audio folds directory")
    parser.add_argument("-s", "--sampling_rate", default=22050, type=int)
    parser.add_argument("-l", "--length", default=4, type=int)
    parser.add_argument("-A","--augments", action="append", help="Augmentation. use as if python script")
    parser.add_argument("-w", "--num_workers", default=4, type=int, help="how many process to perform the augmentations")
    args = parser.parse_args()

    SR = args.sampling_rate
    LENGTH = args.length
    audio_root = args.audio_root

    hdf_raw= h5py.File(os.path.join(audio_root, "%s_%s.hdf5" % ("urbansound8k", SR)), "r")

    # ---- Prepare augmentation ----
    if args.augments is None:
        augment_func_list = []
        augment_name_list = []
    else:
        augment_func_list = list(map(eval, args.augments))
        augment_name_list = []

        # Recover augmentation unique names.
        # If it is not present in the list of augmentation selected by myself, then a new name is created using the
        # initial of the augmentation. They will also be market as inconsistent (meaning different augmentation range
        # could apply
        for augment_str in args.augments:
            if augment_str in reverse_unique_augment:
                augment_name_list.append(reverse_unique_augment[augment_str])
            else:
                augment_func = eval(augment_str)
                unique_name = "I_%s" % augment_func.initial
                augment_name_list.append(unique_name)


    if not augment_func_list:
        print("Nothing to do")
        sys.exit(0)

    # Check if the hdf file for augmentation exist, if yes open it as append
    hdf_path = os.path.join(audio_root, "%s_%s_augmentations.hdf5" % ("urbansound8k", SR))

    if os.path.isfile(hdf_path):
        hdf = h5py.File(hdf_path, 'a')
    else:
        hdf = h5py.File(hdf_path, "w")


    # Utility functions =============================================
    def pad_and_crops(raw):
        if len(raw) < SR * LENGTH:
            missing = (SR * LENGTH) - len(raw)
            raw = np.concatenate((raw, [0] * missing))
        elif len(raw) > SR * LENGTH:
            raw = raw[:SR *LENGTH]

        return raw

    # Compute the augmentation for every file and keep the architecture (folders)
    to_remove = [".DS_Store"]

    for augment_name, augment_func in zip(augment_name_list, augment_func_list):

        for fold in tqdm.tqdm(range(1, 11)):

            # Check if the augmentation exist
            # if yes, add the current augmentation as variante (increase dim 0 by 1)
            # If no, create the new groupe with shape (1, ...)

            # create the group if needed
            folder = "fold%d" % fold
            if folder not in hdf:
                hdf_fold = hdf.create_group(folder)
            else:
                hdf_fold = hdf[folder]

            # recover the list of files from the raw audio hdf file
            fold_filenames = hdf_raw[folder]["filenames"][()]

            # for every raw audio, compute the augmentation
            raw_audios = hdf_raw[folder]["data"][()]

            # create the augment if needed with shape (1, ...), extended the dataset otherwise
            if augment_name not in hdf_fold:
                hdf_fold.create_dataset(augment_name, chunks=True, shape=(1, len(fold_filenames), SR * LENGTH), maxshape=(None, None, SR *LENGTH))
                index_to_store = 0
            else:
                current_shape = hdf_fold[augment_name].shape
                hdf_fold[augment_name].resize(current_shape[0] + 1, axis=0)
                index_to_store = current_shape[0] # <-- already add 1 when using <shape> (like using <len>)

            print("augment_func: ", augment_func)
            print("raw audio len: ", raw_audios.shape)

            pool = Pool(args.num_workers)

            jobs = [pool.apply_async(augment_func, (raw, )) for raw in raw_audios]
            results = [job.get() for job in jobs]

            results = np.asarray(results)

            # write the result of the augmentation in the appropriate augmet variante
            hdf_fold[augment_name][index_to_store, :, :] = results

            # Write the folder filename if not already done
            if "filenames" not in hdf_fold:
                hdf_fold.create_dataset("filenames", (len(fold_filenames), ), dtype=h5py.special_dtype(vlen=str))
                for i in range(len(fold_filenames)):
                    hdf_fold["filenames"][i] = fold_filenames[i]

            # Close the process pool and go to the next folder
            pool.close()
            pool.join()


    hdf.close()
    hdf_raw.close()
