import numpy as np

import torch
import torch.nn as nn
import librosa

from ubs8k.datasetManager import DatasetManager
from ubs8k.layers import ConvPoolReLU, ConvReLU, ConvBNReLUPool, ConvAdvBNReLUPool, Sequential_adv


class cnn(nn.Module):
    def __init__(self, **kwargs):
        super(cnn, self).__init__()

        self.features = nn.Sequential(
            ConvPoolReLU(1, 24, 3, 1, 1, (4, 2), (4, 2)),
            ConvPoolReLU(24, 48, 3, 1, 1, (4, 2), (4, 2)),
            ConvPoolReLU(48, 48, 3, 1, 1, (4, 2), (4, 2)),
            ConvReLU(48, 48, 3, 1, 1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(1008, 10),
            #             nn.ReLU(inplace=True),
            #             nn.Dropout(0.5),
            #             nn.Linear(64, 10),
        )

    def forward(self, x):
        x = x.view(-1, 1, *x.shape[1:])

        x = self.features(x)
        x = self.classifier(x)

        return x


class cnn_advBN(nn.Module):
    """
    Basic CNN model with adversarial dedicated Batch Normalization

    """
    def __init__(self, *kwargs):
        super(cnn_advBN, self).__init__()

        self.features = nn.Sequential(
            ConvAdvBNReLUPool(1, 24, 3, 1, 1, (4, 2), (4, 2)),
            ConvAdvBNReLUPool(24, 48, 3, 1, 1, (4, 2), (4, 2)),
            ConvAdvBNReLUPool(48, 48, 3, 1, 1, (4, 2), (4, 2)),
            ConvReLU(48, 48, 3, 1, 1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(1008, 10),
            #             nn.ReLU(inplace=True),
            #             nn.Dropout(0.5),
            #             nn.Linear(64, 10),
        )

    def forward(self, x):
        x = x.view(-1, 1, *x.shape[1:])

        x = self.features(x)
        x = self.classifier(x)

        return x


class ScalableCnn(nn.Module):
    """
    Compound Scaling based CNN
    see: https://arxiv.org/pdf/1905.11946.pdf
    """

    def __init__(self, dataset: DatasetManager,
                 compound_scales: tuple = (1, 1, 1),
                 initial_conv_inputs=[1, 32, 64, 64],
                 initial_conv_outputs=[32, 64, 64, 64],
                 initial_linear_inputs=[1344, ],
                 initial_linear_outputs=[10, ],
                 initial_resolution=[64, 173],
                 round_up: bool = False,
                 **kwargs
                 ):
        super(ScalableCnn, self).__init__()
        self.compound_scales = compound_scales
        self.dataset = dataset
        round_func = np.floor if not round_up else np.ceil

        alpha, beta, gamma = compound_scales[0], compound_scales[1], compound_scales[2]

        initial_nb_conv = len(initial_conv_inputs)
        initial_nb_dense = len(initial_linear_inputs)

        # Apply compound scaling

        # resolution ----
        # WARNING - RESOLUTION WILL CHANGE THE FEATURES EXTRACTION OF THE SAMPLE
        new_n_mels = int(round_func(initial_resolution[0] * gamma))
        new_n_time_bins = int(round_func(initial_resolution[1] * gamma))
        new_hop_length = int(round_func( (self.dataset.sr * DatasetManager.LENGTH) / new_n_time_bins))

        self.scaled_resolution = (new_n_mels, new_n_time_bins)
        print("new scaled resolution: ", self.scaled_resolution)

        if gamma > 1.0:
            dataset.extract_feature = self.generate_feature_extractor(new_n_mels, new_hop_length)

        # depth ----
        scaled_nb_conv = round_func(initial_nb_conv * alpha)
        scaled_nb_linear = round_func(initial_nb_dense * alpha)

        if scaled_nb_conv != initial_nb_conv:  # Another conv layer must be created
            print("More conv layer must be created")
            gaps = np.array(initial_conv_outputs) - np.array(initial_conv_inputs)  # average filter gap
            avg_gap = gaps.mean()

            while len(initial_conv_inputs) < scaled_nb_conv:
                initial_conv_outputs.append(int(round_func(initial_conv_outputs[-1] + avg_gap)))
                initial_conv_inputs.append(initial_conv_outputs[-2])

            print("new conv layers:")
            print("inputs: ", initial_conv_inputs)
            print("ouputs: ", initial_conv_outputs)

        if scaled_nb_linear != initial_nb_dense:  # Another dense layer must be created
            print("More dense layer must be created")
            dense_list = np.linspace(initial_linear_inputs[0], initial_linear_outputs[-1], scaled_nb_linear + 1)
            initial_linear_inputs = dense_list[:-1]
            initial_linear_outputs = dense_list[1:]

            print("new dense layers:")
            print("inputs: ", initial_linear_inputs)
            print("ouputs: ", initial_linear_outputs)

        # width ----
        scaled_conv_inputs = [int(round_func(i * beta)) for i in initial_conv_inputs]
        scaled_conv_outputs = [int(round_func(i * beta)) for i in initial_conv_outputs]
        scaled_dense_inputs = [int(round_func(i * beta)) for i in initial_linear_inputs]
        scaled_dense_outputs = [int(round_func(i * beta)) for i in initial_linear_outputs]

        # Check how many conv with pooling layer can be used
        nb_max_pooling = np.min([np.log2(self.scaled_resolution[0]), int(np.log2(self.scaled_resolution[1]))])
        nb_model_pooling = len(scaled_conv_inputs)

        if nb_model_pooling > nb_max_pooling:
            nb_model_pooling = nb_max_pooling

        # fixe initial and final conv & linear input
        scaled_conv_inputs[0] = 1
        scaled_dense_inputs[0] = self.calc_initial_dense_input(self.scaled_resolution, nb_model_pooling,
                                                               scaled_conv_outputs)
        scaled_dense_outputs[-1] = 10

        # ======== Create the convolution part ========
        features = []

        # Create the layers
        for idx, (inp, out) in enumerate(zip(scaled_conv_inputs, scaled_conv_outputs)):
            if idx < nb_model_pooling:
                dropout = 0.3 if idx != 0 else 0.0
                features.append(ConvBNReLUPool(inp, out, 3, 1, 1, (2, 2), (2, 2), dropout))

            else:
                features.append(ConvReLU(inp, out, 3, 1, 1))

        self.features = nn.Sequential(
            *features,
        )

        # ======== Craete the classifier part ========
        linears = []
        for inp, out in zip(scaled_dense_inputs[:-1], scaled_dense_outputs[:-1]):
            print(inp, out)
            linears.append(nn.Linear(inp, out))
            linears.append(nn.ReLU6(inplace=True))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            *linears,
            nn.Linear(scaled_dense_inputs[-1], scaled_dense_outputs[-1])
        )

    def forward(self, x):
        x = x.view(-1, 1, *x.shape[1:])

        x = self.features(x)
        x = self.classifier(x)

        return x

    def calc_initial_dense_input(self, resolution, nb_model_pooling, conv_outputs):
        dim1 = resolution[0]
        dim2 = resolution[1]

        for i in range(int(nb_model_pooling)):
            dim1 = dim1 // 2
            dim2 = dim2 // 2

        return dim1 * dim2 * conv_outputs[-1]

    def generate_feature_extractor(self, n_mels, hop_length):
        def extract_feature(raw_data, filename = None, cached = False):
            feat = librosa.feature.melspectrogram(
                raw_data, self.dataset.sr, n_fft=2048, hop_length=hop_length, n_mels=n_mels, fmin=0, fmax=self.dataset.sr // 2)
            feat = librosa.power_to_db(feat, ref=np.max)
            return feat

        return extract_feature


class ScalableCnn_advBN(nn.Module):
    """
    Compound Scaling based CNN
    see: https://arxiv.org/pdf/1905.11946.pdf
    """

    def __init__(self, dataset,
                 compound_scales: tuple = (1, 1, 1),
                 initial_conv_inputs=[1, 32, 64, 64],
                 initial_conv_outputs=[32, 64, 64, 64],
                 initial_linear_inputs=[1344, ],
                 initial_linear_outputs=[10, ],
                 initial_resolution=[64, 173],
                 **kwargs
                 ):
        super(ScalableCnn_advBN, self).__init__()
        alpha, beta, gamma = compound_scales[0], compound_scales[1], compound_scales[2]

        initial_nb_conv = len(initial_conv_inputs)
        initial_nb_dense = len(initial_linear_inputs)

        # Apply compound scaling

        # resolution ----
        # WARNING - RESOLUTION WILL CHANGE THE FEATURES EXTRACTION OF THE SAMPLE
        new_n_mels = int(np.floor(initial_resolution[0] * gamma))
        new_hop_length = int(np.floor(initial_resolution[1] * gamma))
        self.scaled_resolution = (new_n_mels, new_hop_length)
        print("new scaled resolution: ", self.scaled_resolution)

        if gamma > 1.0:
            dataset.extract_feature = self.generate_feature_extractor(new_n_mels, new_hop_length)

        # depth ----
        scaled_nb_conv = np.floor(initial_nb_conv * alpha)
        scaled_nb_linear = np.floor(initial_nb_dense * alpha)

        if scaled_nb_conv != initial_nb_conv:  # Another conv layer must be created
            print("More conv layer must be created")
            gaps = np.array(initial_conv_outputs) - np.array(initial_conv_inputs)  # average filter gap
            avg_gap = gaps.mean()

            while len(initial_conv_inputs) < scaled_nb_conv:
                initial_conv_outputs.append(int(np.floor(initial_conv_outputs[-1] + avg_gap)))
                initial_conv_inputs.append(initial_conv_outputs[-2])

            print("new conv layers:")
            print("inputs: ", initial_conv_inputs)
            print("ouputs: ", initial_conv_outputs)

        if scaled_nb_linear != initial_nb_dense:  # Another dense layer must be created
            print("More dense layer must be created")
            dense_list = np.linspace(initial_linear_inputs[0], initial_linear_outputs[-1], scaled_nb_linear + 1)
            initial_linear_inputs = dense_list[:-1]
            initial_linear_outputs = dense_list[1:]

            print("new dense layers:")
            print("inputs: ", initial_linear_inputs)
            print("ouputs: ", initial_linear_outputs)

        # width ----
        scaled_conv_inputs = [int(np.floor(i * beta)) for i in initial_conv_inputs]
        scaled_conv_outputs = [int(np.floor(i * beta)) for i in initial_conv_outputs]
        scaled_dense_inputs = [int(np.floor(i * beta)) for i in initial_linear_inputs]
        scaled_dense_outputs = [int(np.floor(i * beta)) for i in initial_linear_outputs]

        # Check how many conv with pooling layer can be used
        nb_max_pooling = np.min([np.log2(self.scaled_resolution[0]), int(np.log2(self.scaled_resolution[1]))])
        nb_model_pooling = len(scaled_conv_inputs)

        if nb_model_pooling > nb_max_pooling:
            nb_model_pooling = nb_max_pooling

        # fixe initial and final conv & linear input
        scaled_conv_inputs[0] = 1
        scaled_dense_inputs[0] = self.calc_initial_dense_input(self.scaled_resolution, nb_model_pooling,
                                                               scaled_conv_outputs)
        scaled_dense_outputs[-1] = 10

        # ======== Create the convolution part ========
        features = []

        # Create the layers
        for idx, (inp, out) in enumerate(zip(scaled_conv_inputs, scaled_conv_outputs)):
            if idx < nb_model_pooling:
                dropout = 0.3 if idx != 0 else 0.0
                features.append(ConvAdvBNReLUPool(inp, out, 3, 1, 1, (2, 2), (2, 2), dropout))

            else:
                features.append(ConvReLU(inp, out, 3, 1, 1))

        self.features = Sequential_adv(*features)

        # ======== Craete the classifier part ========
        linears = []
        for inp, out in zip(scaled_dense_inputs[:-1], scaled_dense_outputs[:-1]):
            print(inp, out)
            linears.append(nn.Linear(inp, out))
            linears.append(nn.ReLU6(inplace=True))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            *linears,
            nn.Linear(scaled_dense_inputs[-1], scaled_dense_outputs[-1])
        )

    def forward(self, x, adv: bool = False):
        x = x.view(-1, 1, *x.shape[1:])

        x = self.features(x, adv)
        x = self.classifier(x)

        return x

    def calc_initial_dense_input(self, resolution, nb_model_pooling, conv_outputs):
        dim1 = resolution[0]
        dim2 = resolution[1]

        for i in range(int(nb_model_pooling)):
            dim1 = dim1 // 2
            dim2 = dim2 // 2

        return dim1 * dim2 * conv_outputs[-1]

    def generate_feature_extractor(self, n_mels, hop_length):
        def extract_feature(raw_data):
            feat = librosa.feature.melspectrogram(
                raw_data, self.sr, n_fft=2048, hop_length=hop_length, n_mels=n_mels, fmin=0, fmax=self.sr // 2)
            feat = librosa.power_to_db(feat, ref=np.max)
            return feat

        return extract_feature


def scallable2(**kwargs):
    dataset = kwargs["dataset"]
    parameters = dict(
        dataset=dataset,
        initial_conv_inputs=[1, 44, 89, 89, 89, 111],
        initial_conv_outputs=[44, 89, 89, 89, 111, 133],
        initial_linear_inputs=[266, ],
        initial_linear_outputs=[10, ]
    )

    return ScalableCnn(**parameters)


def scallable1_new(**kwargs):
    """The new scallable 1 model feature resolution scaling.
    It is also used as a starter for the new scallable 2 model
    """
    dataset = kwargs["dataset"]
    compound_scales = kwargs.get("compound_scales", (1.36, 1.0, 1.21))

    parameters = dict(
        dataset=dataset,
        compound_scales = compound_scales,
        initial_conv_inputs=[1, 32, 64, 64],
        initial_conv_outputs=[32, 64, 64, 64],
        initial_linear_inputs=[1344, ],
        initial_linear_outputs=[10, ],
        initial_resolution=[64, 173],
        round_up=True,
    )

    return ScalableCnn(**parameters)

