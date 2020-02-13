import numpy as np


def call_counter(func):
    """ use for initial to count number of time a same augmentation is called in a script.
    Noise called once will return N1
    Noise called Thrice will return N1 N2 N3
    """
    def decorator(*args, **kwargs):
        value = func(*args, **kwargs)

        if value not in call_counter.counter:
            call_counter.counter[value] = 1
        else:
            call_counter.counter[value] += 1

        return value+str(call_counter.counter[value])

    call_counter.counter = dict()
    return decorator


class Augmentation:
    def __init__(self, ratio):
        """
        Args:
            ratio: The probability to apply the augmentation
        """
        self.ratio = ratio

    def _perform_augmentation(self, data):
        """Perform augmentation if needed

        Args:
            data: The spectrogram to use
        """
        if np.random.random() < self.ratio:
            return self._apply(data)
        return data

    def _apply(self, data):
        if len(data.shape) == 2:
            return self.apply_helper(data)

        if len(data.shape) == 3:
            out = data.copy()

            for i in range(len(out)):
                out[i] = self.apply_helper(out[i])

            return out

        else:
            print("Warning, can't be used of more than 3 dimensions, no modification will be apply")
            return data

    def apply_helper(self, data):
        raise NotImplementedError("This is an abstract class")

    def __call__(self, data):
        """
        Args:
            data: The spectrogram to use
        """
        return self._perform_augmentation(data)

    @property
    @call_counter
    def initial(self):
        class_name = self.__class__.__name__

        # Get upper case
        initial = ''.join([c for c in class_name if c.isupper()])
        return initial


class SignalAugmentation(Augmentation):
    def __init__(self, ratio):
        super().__init__(ratio)


class SpecAugmentation(Augmentation):
    def __init__(self, ratio):
        super().__init__(ratio)


if __name__ == '__main__':
    test = SignalAugmentation(1.0)
    print(test.initial)