import librosa
import numpy as np

from .augmentations import SignalAugmentation


class TimeStretch(SignalAugmentation):
    def __init__(self, ratio, rate: tuple = (0.9, 1.1)):
        super().__init__(ratio)
        self.rate = rate

    def apply_helper(self, data):
        rate = np.random.uniform(*self.rate)
        output = librosa.effects.time_stretch(data, rate)
        return output


class PitchShiftRandom(SignalAugmentation):
    def __init__(self, ratio, sampling_rate: int = 22050, steps: tuple = (-3, 3)):
        super().__init__(ratio)
        self.sampling_rate = sampling_rate
        self.steps = steps

    def apply_helper(self, data):
        nb_steps = np.random.uniform(*self.steps)
        output = librosa.effects.pitch_shift(data, sr=self.sampling_rate, n_steps=nb_steps)
        return output


class PitchShiftChoice(SignalAugmentation):
    def __init__(self, ratio, sampling_rate: int = 22050, choice: tuple = (-2, -1, 1, 2)):
        super().__init__(ratio)
        self.sampling_rate = sampling_rate
        self.choice = choice

    def apply_helper(self, data):
        nb_steps = np.random.choice(self.choice)
        output = librosa.effects.pitch_shift(data, sr=self.sampling_rate, n_steps=nb_steps)
        return output


class Level(SignalAugmentation):
    def __init__(self, ratio, rate: tuple = (0.8, 1.2)):
        super().__init__(ratio)
        self.rate = rate

    def apply_helper(self, data):
        rate = np.random.uniform(*self.rate)
        return rate*data


# TODO better implementation
class Noise2(SignalAugmentation):
     def __init__(self, ratio, noise_factor: tuple = (0.1, 0.4)):
         super().__init__(ratio)
         self.noise_factor = noise_factor

     def apply_helper(self, data):
         noise = np.random.randn(len(data))
         noise_factor = np.random.uniform(*self.noise_factor)
         return data + noise_factor * noise


class Noise(SignalAugmentation):
    def __init__(self, ratio, target_snr: int = 10):
        super().__init__(ratio)
        self.target_snr = target_snr

    def apply_helper(self, data):
        # calculate noise in signal
        P_data = data**2
        average_noise_db = 10 * np.log10(P_data.mean())

        # Adjust target SNR
        t_snr = average_noise_db - self.target_snr

        # calc scale factor
        k = 10 ** (t_snr / 10)

        # noise
        noise = np.random.normal(0, np.sqrt(k), size=len(data))

        return data + noise


class Occlusion(SignalAugmentation):
    def __init__(self, ratio, sampling_rate: int = 22050, max_size: float = 1):
        super().__init__(ratio)
        self.max_size = max_size
        self.sampling_rate = sampling_rate

    def apply_helper(self, data):
        max_occlu_size = self.sampling_rate * self.max_size
        if max_occlu_size > len(data):
            max_occlu_size = len(data) // 4

        occlu_size = np.random.randint(0, max_occlu_size)
        occlu_pos = np.random.randint(0, len(data) - occlu_size)

        cp_data = data.copy()
        cp_data[occlu_pos:occlu_pos + occlu_size] = 0

        return cp_data


class Clip(SignalAugmentation):
    def __init__(self, ratio, range: tuple = (-0.9, 0.9)):
        super().__init__(ratio)
        self.range = range

    def apply_helper(self, data):
        return np.clip(data, *range)


if __name__ == '__main__':
    ts = TimeStretch(0.01, rate=(0.9, 1.1))
    ps = PitchShiftRandom(0.01, 8000, steps=(-2, 2))
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