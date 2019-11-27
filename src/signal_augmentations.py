import numpy as np
import librosa


class Augmentation:
    def __init__(self, ratio):
        self.ratio = ratio

    def _perform_augmentation(self, data):
        """Perform augmentation if needed"""
        if np.random.random() < self.ratio:
            return self._apply(data)
        return data

    def _apply(self, data):
        raise NotImplementedError("This is an abstract class")

    def __call__(self, data):
        return self._perform_augmentation(data)


class TimeStretch(Augmentation):
    def __init__(self, ratio, rate: tuple = (0.9, 1.1)):
        super().__init__(ratio)
        self.rate = rate

    def _apply(self, data):
        rate = np.random.uniform(*self.rate)
        output = librosa.effects.time_stretch(data, rate)
        return output


class PitchShift(Augmentation):
    def __init__(self, ratio, sampling_rate: int, steps: tuple = (-3, 3)):
        super().__init__(ratio)
        self.sr = sampling_rate
        self.steps = steps

    def _apply(self, data):
        nb_steps = np.random.uniform(*self.steps)
        output = librosa.effects.pitch_shift(data, sr=self.sr, n_steps=nb_steps)
        return output


class Level(Augmentation):
    def __init__(self, ratio, rate: tuple = (0.8, 1.2)):
        super().__init__(ratio)
        self.rate = rate

    def _apply(self, data):
        rate = np.random.uniform(*self.rate)
        return rate*data


# TODO better implementation
class Noise(Augmentation):
    def __init__(self, ratio, noise_factor: tuple = (0.1, 0.4)):
        super().__init__(ratio)
        self.noise_factor = noise_factor

    def _apply(self, data):
        noise = np.random.randn(len(data))
        noise_factor = np.random.uniform(*self.noise_factor)
        return data + noise_factor * noise


class Occlusion(Augmentation):
    def __init__(self, ratio, max_size: float, sampling_rate: int):
        super().__init__(ratio)
        self.max_size = max_size
        self.sampling_rate = sampling_rate

    def _apply(self, data):
        occlu_size = np.random.randint(0, int(self.sampling_rate * self.max_size))
        occlu_pos = np.random.randint(0, len(data) - occlu_size)

        cp_data = data.copy()
        cp_data[occlu_pos:occlu_pos+occlu_size] = 0

        return cp_data


if __name__ == '__main__':
    ts = TimeStretch(0.01, rate=(0.9, 1.1))
    ps = PitchShift(0.01, 8000, steps=(-2, 2))
    l = Level(0.01, rate=(0.8, 1.2))
    n = Noise(0.01, noise_factor=(0.1, 0.4))
    o = Occlusion(0.01, 0.5, 8000)

    augment_func = [ts, ps, l, n, o]

    fake_signal = np.random.uniform(-1, 1, size=(8000*5))

    def basic_stat(signal):
        msg = "%4f - %4f - %4f" % (signal.mean(), signal.var(), signal.std())
        print(msg)

    basic_stat(fake_signal)
    for f in augment_func:
        n_signal = f(fake_signal)
        basic_stat(n_signal)