from dataclasses import dataclass

import librosa
import torch
import torchaudio
from torch import nn


@dataclass
class MelSpectrogramConfig:
    sr: int = 22050
    win_length: int = 1024
    hop_length: int = 256
    n_fft: int = 1024
    f_min: int = 0
    f_max: int = 8000
    n_mels: int = 80
    power: float = 1.0

    # value of melspectrograms if we fed a silence into `MelSpectrogram`
    pad_value: float = -11.5129251


class MelSpectrogram(nn.Module):

    def __init__(self, config: MelSpectrogramConfig):
        super(MelSpectrogram, self).__init__()

        self.config = config

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sr,
            win_length=config.win_length,
            hop_length=config.hop_length,
            n_fft=config.n_fft,
            f_min=config.f_min,
            f_max=config.f_max,
            n_mels=config.n_mels
        )

        # The is no way to set power in constructor in 0.5.0 version.
        self.mel_spectrogram.spectrogram.power = config.power

        # Default `torchaudio` mel basis uses HTK formula. In order to be compatible with WaveGlow
        # we decided to use Slaney one instead (as well as `librosa` does by default).
        mel_basis = librosa.filters.mel(
            sr=config.sr,
            n_fft=config.n_fft,
            n_mels=config.n_mels,
            fmin=config.f_min,
            fmax=config.f_max
        ).T
        self.mel_spectrogram.mel_scale.fb.copy_(torch.tensor(mel_basis))

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        :param audio: Expected shape is [B, T]
        :return: Shape is [B, n_mels, T']
        """

        mel = self.mel_spectrogram(audio) \
            .clamp_(min=1e-5) \
            .log_()
        return mel


@dataclass
class TrainConfig:
    upsampling_coefs = [512, 8, 8, 2, 2]
    kernel_global_lib = [3, 7, 11]
    dilations_global_lib = [[1, 3, 5] for _ in range(3)]
    wave_length = 8192
    wav2mel = MelSpectrogram(MelSpectrogramConfig()).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    save_dir = "./model_new"
    logger_path = "./logger"
    mel_ground_truth = "./mels"
    data_path = './data/train.txt'
    wave_path = './data/LJSpeech-1.1/wavs'
    audio_path = "./audio"
    fake_path = "./fake_audio"
    n_mels = 80

    visualize = 'wandb'
    wandb_project = 'Neural Vocoder'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = 'cuda:0'

    batch_size = 1
    epochs = 2000

    lr = 2e-4
    weight_decay = 1e-2
    grad_clip_thresh = 1.0

    batch_expand_size = 16
    segment_size = 1024 * 8
