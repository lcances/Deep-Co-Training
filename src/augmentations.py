import numpy as np


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

    @property
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