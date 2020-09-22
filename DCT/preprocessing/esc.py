from torch.nn import Sequential
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB

transforms = Sequential(
    MelSpectrogram(sample_rate=44100, n_fft=2048, hop_length=512, n_mels=64),
    AmplitudeToDB(),
)
