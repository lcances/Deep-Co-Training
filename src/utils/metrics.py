import torch

class Metrics:
    def __init__(self, epsilon=1e-10):
        self.value = 0
        self.accumulate_value = 0
        self.count = 0
        self.epsilon = epsilon

    def reset(self):
        self.accumulate_value = 0
        self.count = 0

    def __call__(self):
        self.count += 1


class BinaryAccuracy(Metrics):
    def __init__(self, epsilon=1e-10):
        Metrics.__init__(self, epsilon)

    def __call__(self, y_pred, y_true):
        super().__call__()

        with torch.set_grad_enabled(False):
            y_pred = (y_pred > 0.5).float()
            correct = (y_pred == y_true).float().sum()
            self.value = correct / (y_true.shape[0] * y_true.shape[1])

            self.accumulate_value += self.value
            return self.accumulate_value / self.count


class CategoricalAccuracy(Metrics):
    def __init__(self, epsilon=1e-10):
        Metrics.__init__(self, epsilon)

    def __call__(self, y_pred, y_true):
        super().__call__()

        with torch.set_grad_enabled(False):
            self.value = torch.mean((y_true == y_pred).float())

            self.accumulate_value += self.value

            return self.accumulate_value / self.count


class Ratio(Metrics):
    def __init__(self, epsilon=1e-10):
        Metrics.__init__(self, epsilon)

    def __call__(self, y_pred, y_adv_pred):
        super().__call__()

        results = zip(y_pred, y_adv_pred)
        results_bool = [int(r[0] != r[1]) for r in results]
        self.value = sum(results_bool) / len(results_bool) * 100
        self.accumulate_value += self.value

        return self.accumulate_value / self.count

