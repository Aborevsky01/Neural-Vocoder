import logging
import os
import random
import shutil
import time

import librosa
import seaborn as sns
import torch
from speechbrain.utils.data_utils import download_file
from torch.utils.data import Dataset
from tqdm import tqdm

from utils import ROOT_PATH

sns.set()

import sys

sys.path.append('.')

logger = logging.getLogger(__name__)

URL_LINKS = {
    "dataset": "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2",
}


def check_dataset_loaded():
    data_dir = ROOT_PATH / "data"
    if not data_dir.exists():
        arch_path = data_dir / "LJSpeech-1.1.tar.bz2"
        print(f"Loading LJSpeech")
        download_file(URL_LINKS["dataset"], arch_path)
        shutil.unpack_archive(arch_path, data_dir)


def get_data_to_buffer(train_config):
    buffer = list()
    wave_names = sorted(os.listdir(train_config.wave_path))

    start = time.perf_counter()
    for i in tqdm(range(len(wave_names))):
        wave_name = os.path.join(train_config.wave_path, wave_names[i])
        wave = torch.from_numpy(librosa.load(wave_name)[0]).unsqueeze(0)
        start_pos = random.randint(0, wave.shape[-1] - train_config.segment)
        wave = wave[:, start_pos: train_config.segment + start_pos]

        mel_gt_path = ROOT_PATH / 'mels' / "ljspeech-mel-{0}.npy".format(i + 1)
        if not mel_gt_path.exists():
            mel_gt_target = train_config.wav2mel(wave)
            torch.save(mel_gt_target, mel_gt_path)
        else:
            mel_gt_target = torch.load(mel_gt_path)
        buffer.append({"mel": mel_gt_target, "wave": wave})

    end = time.perf_counter()
    print("cost {:.2f}s to load all data into buffer.".format(end - start))

    return buffer


class BufferDataset(Dataset):
    def __init__(self, buffer):
        self.buffer = buffer
        self.length_dataset = len(self.buffer)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, idx):
        return self.buffer[idx]


class EvalDataset(Dataset):
    def __init__(self, train_config):
        self.buffer = []
        for path, dirnames, filenames in os.walk(train_config.audio_path):
            for file in filenames:
                wav = torch.from_numpy(librosa.load(os.path.join(train_config.audio_path, file))[0]).unsqueeze(0)
                mel = train_config.wav2mel(wav.to(train_config.device)).squeeze()
                self.buffer.append({'waves': wav, 'mel': mel})

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        return self.buffer[idx]
