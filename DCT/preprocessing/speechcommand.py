import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB


class PadUpTo(nn.Module):
    def __init__(self, target_length, mode: str = "constant", value: int = 0):
        super().__init__()
        self.target_length = target_length
        self.mode = mode
        self.value = value

    def forward(self, x):
        actual_length = x.size()[-1]
        return F.pad(input=x, pad=(0, (self.target_length - actual_length)),
                     mode=self.mode, value=self.value)


transforms = nn.Sequential(
    PadUpTo(target_length=16000, mode="constant", value=0),
    MelSpectrogram(sample_rate=16000, n_fft=2048, hop_length=512, n_mels=64),
    AmplitudeToDB(),
)
