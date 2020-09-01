import numpy as np

def linear_rampup(current_epoch, ramp_length):
    if ramp_length == 0:
        return 0.0
    
    if current_epoch >= ramp_length:
        return 1.0

    return current_epoch / ramp_length

def cosine_rampup(current_epoch, ramp_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    if ramp_length == 0:
        return 0.0
    
    if current_epoch >= ramp_length:
        return 1.0
    
    return - float(.5 * (np.cos(np.pi * current_epoch / ramp_length) - 1))

def sigmoid_rampup(current_epoch, ramp_length):
    """
    https://arxiv.org/pdf/1803.05984.pdf
    Exponential rampup from https://arxiv.org/abs/1610.02242
    """
    if ramp_length == 0:
        return 0.0
    
    if current_epoch >= ramp_length:
        return 1.0

    current = np.clip(current_epoch, 0.0, ramp_length)
    phase = 1.0 - current / ramp_length
    return float(np.exp(-5.0 * phase * phase))
    
def sigmoid_rampdown(current_epoch, ramp_length):
    if ramp_length == 0:
        return 1.0
    
    if current_epoch >= ramp_length:
        return 0.0

    current = np.clip(current_epoch, 0.0, ramp_length)
    phase = 1.0 - (current / ramp_length)
    return 1 - float(np.exp(-5.0 * phase**2))


class Warmup:
    def __init__(self, max, nb_epoch, method):
        self.max = max
        self.nb_epoch = nb_epoch
        self.method = method
        self.current_epoch = 0
        self.value = method(0, nb_epoch)

    def reset(self):
        self.current_epoch = 0
        self.value = method(0, nb_epoch)

    def step(self):
        if self.current_epoch < self.nb_epoch:
            self.current_epoch += 1
            ramp = self.method(self.current_epoch, self.nb_epoch)
            self.value = self.max * ramp

        return self.value

    def __call__(self):
        return self.value

if __name__ == '__main__':
    linear_warmup = Warmup(10, 80, linear_rampup)
    exp_warmup = Warmup(10, 80, sigmoid_rampup)
    cosine_warmup = Warmup(10, 80, cosine_rampup)


    linear_values = []
    exp_values = []
    cosine_values = []

    for _ in range(150):
        linear_values.append(linear_warmup.value)
        exp_values.append(exp_warmup.value)
        cosine_values.append(cosine_warmup.value)

        linear_warmup.step()
        exp_warmup.step()
        cosine_warmup.step()

    import matplotlib.pyplot as plt

    plt.figure(0)
    plt.subplot(3, 1, 1)
    plt.plot(linear_values, label="linear")
    plt.plot(exp_values, label="exponential")
    plt.plot(cosine_values, label="sigmoid")
    plt.legend()
    plt.show()
