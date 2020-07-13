import numpy as np

import torch.nn as nn
import librosa

from ubs8k.datasetManager import DatasetManager, conditional_cache_v2
from .layers import ConvPoolReLU, ConvReLU, ConvBNReLUPool, ConvAdvBNReLUPool


class cnn(nn.Module):
    def __init__(self, **kwargs):
        nn.Module.__init__(self)

        self.features = nn.Sequential(
            ConvPoolReLU(1, 32, 3, 1, 1, pool_kernel_size=(4, 2), pool_stride=(4, 2), dropout=0.0),
            ConvPoolReLU(32, 32, 3, 1, 1, pool_kernel_size=(4, 2), pool_stride=(4, 2), dropout=0.3),
            ConvPoolReLU(32, 32, 3, 1, 1, pool_kernel_size=(4, 2), pool_stride=(4, 2), dropout=0.3),
            nn.Conv2d(32, 32, 1, 1, 0),
            nn.ReLU6(),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(1696, 10) # TODO fill
        )

    def forward(self, x):
        x = x.view(-1, 1, *x.shape[1:])

        x = self.features(x)
        x = self.classifier(x)

        return x


class cnn0(nn.Module):
    def __init__(self, **kwargs):
        super(cnn0, self).__init__()

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

    def __init__(self, manager: DatasetManager,
                 compound_scales: tuple = (1, 1, 1),
                 phi: float = 1.0,
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
        self.manager = manager
        self.round_func = np.floor if not round_up else np.ceil

        alpha, beta, gamma = compound_scales[0], compound_scales[1], compound_scales[2]
        alpha, beta, gamma = alpha**phi, beta**phi, gamma**phi

        initial_nb_conv = len(initial_conv_inputs)
        initial_nb_dense = len(initial_linear_inputs)

        # Apply compound scaling

        # resolution ----
        # WARNING - RESOLUTION WILL CHANGE THE FEATURES EXTRACTION OF THE SAMPLE
        new_n_mels = int(self.round_func(initial_resolution[0] * gamma))
        new_n_time_bins = int(self.round_func(initial_resolution[1] * gamma))
        new_hop_length = int(self.round_func( (self.manager.sr * DatasetManager.LENGTH) / new_n_time_bins))

        self.scaled_resolution = (new_n_mels, new_n_time_bins)
        print("new scaled resolution: ", self.scaled_resolution)

        self.manager.extract_feature = self.generate_feature_extractor(new_n_mels, new_hop_length)

        # ======== CONVOLUTION PARTS ========
        # ---- depth ----
        scaled_nb_conv = self.round_func(initial_nb_conv * alpha)
        
        new_conv_inputs, new_conv_outputs = initial_conv_inputs.copy(), initial_conv_outputs.copy()
        if scaled_nb_conv != initial_nb_conv:  # Another conv layer must be created
            print("More conv layer must be created")
            gaps = np.array(initial_conv_outputs) - np.array(initial_conv_inputs)  # average filter gap
            avg_gap = gaps.mean()

            while len(new_conv_inputs) < scaled_nb_conv:
                new_conv_outputs.append(int(self.round_func(new_conv_outputs[-1] + avg_gap)))
                new_conv_inputs.append(new_conv_outputs[-2])
        
        # ---- width ----
        scaled_conv_inputs = [int(self.round_func(i * beta)) for i in new_conv_inputs]
        scaled_conv_outputs = [int(self.round_func(i * beta)) for i in new_conv_outputs]
        
        print("new conv layers:")
        print("inputs: ", scaled_conv_inputs)
        print("ouputs: ", scaled_conv_outputs)
        
        # Check how many conv with pooling layer can be used
        nb_max_pooling = int(np.floor(np.min([np.log2(self.scaled_resolution[0]), int(np.log2(self.scaled_resolution[1]))])))
        nb_model_pooling = len(scaled_conv_inputs)

        if nb_model_pooling > nb_max_pooling:
            nb_model_pooling = nb_max_pooling
            
        # fixe initial conv layers
        scaled_conv_inputs[0] = 1
        
        # ======== LINEAR PARTS ========
        # adjust the first dense input with the last convolutional layers
        initial_linear_inputs[0] = self.calc_initial_dense_input(
            self.scaled_resolution,
            nb_model_pooling,
            scaled_conv_outputs
        )
        
        # --- depth ---
        scaled_nb_linear = self.round_func(initial_nb_dense * alpha)
        
        if scaled_nb_linear != initial_nb_dense:  # Another dense layer must be created
            print("More dense layer must be created")
            dense_list = np.linspace(initial_linear_inputs[0], initial_linear_outputs[-1], scaled_nb_linear + 1)
            initial_linear_inputs = dense_list[:-1]
            initial_linear_outputs = dense_list[1:]
            
        # --- width ---
        scaled_dense_inputs = [int(self.round_func(i * beta)) for i in initial_linear_inputs]
        scaled_dense_outputs = [int(self.round_func(i * beta)) for i in initial_linear_outputs]
        
        # fix first and final linear layer
        scaled_dense_inputs[0] = self.calc_initial_dense_input(self.scaled_resolution,
                                                                nb_model_pooling,
                                                                scaled_conv_outputs)
        scaled_dense_outputs[-1] = 10
        
        print("new dense layers:")
        print("inputs: ", scaled_dense_inputs)
        print("ouputs: ", scaled_dense_outputs)

        # ======== BUILD THE MODEL=========
        # features part ----
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

        # classifier part ----
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
            dim1 = int(self.round_func(dim1 / 2))
            dim2 = int(self.round_func(dim2 / 2))

        return dim1 * dim2 * conv_outputs[-1]
    
    def generate_feature_extractor(self, n_mels, hop_length):
        @conditional_cache_v2
        def extract_feature(raw_data, **kwargs):
            """
            extract the feature for the model. Cache behaviour is implemented with the two parameters filename and cached
            :param raw_data: to audio to transform
            :key key: Unique key link to the raw_data, used internally by the cache system
            :key cached: use or not the cache system
            :return: the feature extracted from the raw audio
            """
            feat = librosa.feature.melspectrogram(
                raw_data, self.manager.sr, n_fft=2048, hop_length=hop_length, n_mels=n_mels, fmin=0, fmax=self.manager.sr // 2)
            feat = librosa.power_to_db(feat, ref=np.max)
            
            return feat
        
        print("new feature extraction function generation: hop_length = %s" % hop_length)
        return extract_feature


def scallable1(manager):
    alpha, beta, gamma = 1.357143, 1.214286, 1.000000
    phi = 2.2

    parameters = dict(
        manager=manager,
        compound_scales=(alpha, beta, gamma),
        phi=phi,
        
        initial_conv_inputs=[1, 24, 48, 48],
        initial_conv_outputs=[24, 48, 48, 48],
        initial_linear_inputs=[720, ],
        initial_linear_outputs=[10, ],
        initial_resolution=[64, 173],
        round_up = False,
    )

    return ScalableCnn(**parameters)
