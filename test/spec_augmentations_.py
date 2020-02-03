import sys

sys.path.append("../src")

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
test_melspectro = dataset.extract_feature(test_raw_audio)

# FractalTimeStretch
def test_FractalTimeStretch():
    from spec_augmentations import FractalTimeStretch
    fts = FractalTimeStretch(1.0, intra_ratio=0.3, min_chunk_size=10, max_chunk_size=25)

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
    from spec_augmentations import FractalFreqStretch
    fts = FractalFreqStretch(1.0, intra_ratio=1, min_chunk_size=20, max_chunk_size=40)

    fts_melspectro = fts(test_melspectro)

    plt.figure(0, figsize=(20, 15))
    plt.subplot(3, 1, 1)
    plt.matshow(test_melspectro, fignum=0)
    plt.subplot(3, 1, 2)
    plt.matshow(fts_melspectro, fignum=0)
    plt.subplot(3, 1, 3)
    plt.matshow(test_melspectro - fts_melspectro, fignum=0)
    plt.show()


# FractalTimeDropout
def test_FractalTimeDropout():
    from spec_augmentations import FractalTimeDropout
    fts = FractalTimeDropout(1.0, intra_ratio=0.1, min_chunk_size=10, max_chunk_size=40)

    fts_melspectro = fts(test_melspectro)

    plt.figure(0, figsize=(20, 15))
    plt.subplot(3, 1, 1)
    plt.matshow(test_melspectro, fignum=0)
    plt.subplot(3, 1, 2)
    plt.matshow(fts_melspectro, fignum=0)
    plt.subplot(3, 1, 3)
    plt.matshow(test_melspectro - fts_melspectro, fignum=0)
    plt.show()


# FractalFreqDropout
def test_FractalFreqDropout():
    from spec_augmentations import FractalFrecDropout
    fts = FractalFrecDropout(1.0, intra_ratio=0.1, min_chunk_size=4, max_chunk_size=10)

    fts_melspectro = fts(test_melspectro)

    plt.figure(0, figsize=(20, 15))
    plt.subplot(3, 1, 1)
    plt.matshow(test_melspectro, fignum=0)
    plt.subplot(3, 1, 2)
    plt.matshow(fts_melspectro, fignum=0)
    plt.subplot(3, 1, 3)
    plt.matshow(test_melspectro - fts_melspectro, fignum=0)
    plt.show()


# FractalFreqDropout + FractalTimeDropout
def test_FractalTimeFreqDropout():
    from spec_augmentations import FractalFrecDropout, FractalTimeDropout
    ffd = FractalFrecDropout(1.0, intra_ratio=0.1, min_chunk_size=10, max_chunk_size=40)
    ftd = FractalTimeDropout(1.0, intra_ratio=0.1, min_chunk_size=4, max_chunk_size=10)
    print(test_melspectro.dtype)

    _melspectro = ffd(test_melspectro)
    _melspectro = ftd(_melspectro)
    print("mel spectro type")
    print(type(_melspectro))
    print(_melspectro.dtype)

    plt.figure(0, figsize=(20, 15))
    plt.subplot(3, 1, 1)
    plt.matshow(test_melspectro, fignum=0)
    plt.subplot(3, 1, 2)
    plt.matshow(_melspectro, fignum=0)
    plt.subplot(3, 1, 3)
    plt.matshow(test_melspectro - _melspectro, fignum=0)
    plt.show()

# randomTimeDropout
def test_RandomTimeDropout():
    from spec_augmentations import RandomTimeDropout
    fts = RandomTimeDropout(1.0, dropout=0.1)

    fts_melspectro = fts(test_melspectro)

    plt.figure(0, figsize=(20, 15))
    plt.subplot(3, 1, 1)
    plt.matshow(test_melspectro, fignum=0)
    plt.subplot(3, 1, 2)
    plt.matshow(fts_melspectro, fignum=0)
    plt.subplot(3, 1, 3)
    plt.matshow(test_melspectro - fts_melspectro, fignum=0)
    plt.show()

# randomFreqDropout
def test_RandomFreqDropout():
    from spec_augmentations import RandomFreqDropout
    fts = RandomFreqDropout(1.0, dropout=0.1)

    fts_melspectro = fts(test_melspectro)

    plt.figure(0, figsize=(20, 15))
    plt.subplot(3, 1, 1)
    plt.matshow(test_melspectro, fignum=0)
    plt.subplot(3, 1, 2)
    plt.matshow(fts_melspectro, fignum=0)
    plt.subplot(3, 1, 3)
    plt.matshow(test_melspectro - fts_melspectro, fignum=0)
    plt.show()

def test_noise():
    from spec_augmentations import Noise
    fts= Noise(1.0, 5)

    fts_melspectro = fts(test_melspectro)

    plt.figure(0, figsize=(20, 15))
    plt.subplot(3, 1, 1)
    plt.matshow(test_melspectro, fignum=0)
    plt.colorbar()
    plt.subplot(3, 1, 2)
    plt.matshow(fts_melspectro, fignum=0)
    plt.colorbar()
    plt.subplot(3, 1, 3)
    plt.matshow(test_melspectro - fts_melspectro, fignum=0)

    plt.tight_layout()
    plt.show()



test_noise()
