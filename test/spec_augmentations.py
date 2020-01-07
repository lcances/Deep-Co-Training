from datasetManager import DatasetManager
import matplotlib.pyplot as plt
import numpy as np
import time

np.random.seed(int(time.time()))

audio_root = "../dataset/audio"
metadata_root = "../dataset/metadata"
dataset = DatasetManager(metadata_root, audio_root, train_fold=[1], val_fold=[])

test_file = list(dataset.audio["train"].keys())[0]
test_raw_audio = dataset.audio["train"][test_file]
print(type(test_raw_audio))
test_melspectro = dataset.extract_feature(test_raw_audio, 22050)

# FractalTimeStretch
def test_FractalTimeStretch():
    from src.spec_augmentations import FractalTimeStretch
    fts = FractalTimeStretch(1.0, intra_ratio=0.3, min_column_size=10, max_column_size=25)

    fts_melspectro = fts(test_melspectro)

    plt.figure(0, figsize=(20, 15))
    plt.subplot(3, 1, 1)
    plt.matshow(test_melspectro, fignum=0)
    plt.subplot(3, 1, 2)
    plt.matshow(fts_melspectro, fignum=0)
    plt.subplot(3, 1, 3)
    plt.matshow(test_melspectro - fts_melspectro, fignum=0)
    plt.show()


# FractalFreqStretch
def test_FractalFreqStretch():
    from src.spec_augmentations import FractalFreqStretch
    fts = FractalFreqStretch(1.0, intra_ratio=1, min_column_size=20, max_column_size=40)

    fts_melspectro = fts(test_melspectro)

    plt.figure(0, figsize=(20, 15))
    plt.subplot(3, 1, 1)
    plt.matshow(test_melspectro, fignum=0)
    plt.subplot(3, 1, 2)
    plt.matshow(fts_melspectro, fignum=0)
    plt.subplot(3, 1, 3)
    plt.matshow(test_melspectro - fts_melspectro, fignum=0)
    plt.show()


# test_FractalTimeStretch()
test_FractalFreqStretch()
