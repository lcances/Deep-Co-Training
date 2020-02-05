from datasetManager import DatasetManager
import matplotlib.pyplot as plt
import numpy as np
import time
import librosa.display

def padding_and_cropping(aug_signal):
    if len(aug_signal) < 4 * 22050:
        missing = (4 * 22050) - len(aug_signal)
        aug_signal = np.concatenate((aug_signal, [0] * missing))

    if len(aug_signal) > 4 * 22050:
        aug_signal = aug_signal[:4 * 22050]

    return aug_signal


if __name__ == '__main__':

    np.random.seed(int(time.time()))

    audio_root = "../dataset/audio"
    metadata_root = "../dataset/metadata"
    dataset = DatasetManager(metadata_root, audio_root, train_fold=[1], val_fold=[])

    test_file = list(dataset.audio["train"].keys())[5]
    test_raw_audio = dataset.audio["train"][test_file]
    print(type(test_raw_audio))
    test_melspectro = dataset.extract_feature(test_raw_audio)

    def test_Noise_():
        from src.signal_augmentations import Noise2
        n = Noise2(1.0)

        noisy_signal = n(test_raw_audio)
        noisy_mel = dataset.extract_feature(noisy_signal)

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
        noisy_mel = dataset.extract_feature(noisy_signal)

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

    def test_TimeStretch():
        from src.signal_augmentations import TimeStretch

        aug = TimeStretch(1.0, rate=(0.5, 1.5))

        aug_signal = aug(test_raw_audio)
        aug_signal = padding_and_cropping(aug_signal)

        aug_mel = dataset.extract_feature(aug_signal)

        plt.figure(0, figsize=(20, 15))
        plt.subplot(3, 1, 1)
        librosa.display.specshow(test_melspectro)
        plt.clim(vmin=-80, vmax=0)
        plt.subplot(3, 1, 2)
        librosa.display.specshow(aug_mel)
        plt.clim(vmin=-80, vmax=0)
        plt.subplot(3, 1, 3)
        librosa.display.specshow(aug_mel - test_melspectro)
        plt.clim(vmin=-80, vmax=0)
        plt.show()

    def test_PitchShiftRandom():
        from src.signal_augmentations import PitchShiftRandom

        aug = PitchShiftRandom(1.0, steps=(-2, 2))

        aug_signal = aug(test_raw_audio)
        aug_signal = padding_and_cropping(aug_signal)

        aug_mel = dataset.extract_feature(aug_signal)

        plt.figure(0, figsize=(20, 15))
        plt.subplot(3, 1, 1)
        librosa.display.specshow(test_melspectro)
        plt.clim(vmin=-80, vmax=0)
        plt.subplot(3, 1, 2)
        librosa.display.specshow(aug_mel)
        plt.clim(vmin=-80, vmax=0)
        plt.subplot(3, 1, 3)
        librosa.display.specshow(aug_mel - test_melspectro)
        plt.clim(vmin=-80, vmax=0)
        plt.show()

    def test_PitchShiftChoice():
        from src.signal_augmentations import PitchShiftChoice

        aug = PitchShiftChoice(1.0, choice=(-2, -1, 1, 2))

        aug_signal = aug(test_raw_audio)
        aug_signal = padding_and_cropping(aug_signal)

        aug_mel = dataset.extract_feature(aug_signal)

        plt.figure(0, figsize=(20, 15))
        plt.subplot(3, 1, 1)
        librosa.display.specshow(test_melspectro)
        plt.clim(vmin=-80, vmax=0)
        plt.subplot(3, 1, 2)
        librosa.display.specshow(aug_mel)
        plt.clim(vmin=-80, vmax=0)
        plt.subplot(3, 1, 3)
        librosa.display.specshow(aug_mel - test_melspectro)
        plt.clim(vmin=-80, vmax=0)
        plt.show()

    def test_Occlusion():
        from src.signal_augmentations import Occlusion

        aug = Occlusion(1.0, max_size=1)

        aug_signal = aug(test_raw_audio)
        aug_signal = padding_and_cropping(aug_signal)

        aug_mel = dataset.extract_feature(aug_signal)

        plt.figure(0, figsize=(20, 15))
        plt.subplot(3, 1, 1)
        librosa.display.specshow(test_melspectro)
        plt.clim(vmin=-80, vmax=0)
        plt.subplot(3, 1, 2)
        librosa.display.specshow(aug_mel)
        plt.clim(vmin=-80, vmax=0)
        plt.subplot(3, 1, 3)
        librosa.display.specshow(aug_mel - test_melspectro)
        plt.clim(vmin=-80, vmax=0)
        plt.show()


    test_Occlusion()
