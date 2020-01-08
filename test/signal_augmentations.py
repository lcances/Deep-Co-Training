from datasetManager import DatasetManager
import matplotlib.pyplot as plt
import numpy as np
import time
import librosa.display

np.random.seed(int(time.time()))

audio_root = "../dataset/audio"
metadata_root = "../dataset/metadata"
dataset = DatasetManager(metadata_root, audio_root, train_fold=[1], val_fold=[])

test_file = list(dataset.audio["train"].keys())[5]
test_raw_audio = dataset.audio["train"][test_file]
print(type(test_raw_audio))
test_melspectro = dataset.extract_feature(test_raw_audio, 22050)

def test_Noise_():
    from src.signal_augmentations import Noise2
    n = Noise2(1.0)

    noisy_signal = n(test_raw_audio)
    noisy_mel = dataset.extract_feature(noisy_signal, 22050)

    plt.figure(0, figsize=(20, 15))
    plt.subplot(2, 1, 1)
    librosa.display.specshow(test_melspectro)
    plt.clim(vmin=-80, vmax=0)
    plt.colorbar()
    plt.subplot(2, 1, 2)
    librosa.display.specshow(noisy_mel)
    plt.clim(vmin=-80, vmax=0)
    plt.colorbar()
    plt.show()

def test_Noise(target_snr: int = 10):
    from src.signal_augmentations import Noise
    n = Noise(1.0, target_snr=target_snr)

    noisy_signal = n(test_raw_audio)
    noisy_mel = dataset.extract_feature(noisy_signal, 22050)

    plt.figure(0, figsize=(20, 15))
    plt.subplot(2, 1, 1)
    librosa.display.specshow(test_melspectro)
    plt.clim(vmin=-80, vmax=0)
    plt.colorbar()
    plt.subplot(2, 1, 2)
    librosa.display.specshow(noisy_mel)
    plt.clim(vmin=-80, vmax=0)
    plt.colorbar()
    plt.show()
    
test_Noise_()
test_Noise(50)

