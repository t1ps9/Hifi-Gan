from src.datasets.base_dataset import BaseDataset
from pathlib import Path
import torchaudio
from tqdm import tqdm
import os
import shutil
import requests
import json
from src.datasets.melspectogram import MelSpectrogram, MelSpectrogramConfig
import wget
import random
import torch

URL_LINK = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"


class LJSpeechDataset(BaseDataset):
    def __init__(self, data_dir=None, wav_max_len=None, device="cuda", *args, **kwargs):
        if data_dir is None:
            data_dir = Path("data/datasets/ljspeech")
            data_dir.mkdir(exist_ok=True, parents=True)
        self.data_dir = data_dir
        self.index = self._load_or_create_index()
        self.wav_max_len = wav_max_len
        self.device = torch.device(device)
        self.mel_spectrogram = MelSpectrogram(MelSpectrogramConfig()).to(self.device)

        super().__init__(self.index, *args, **kwargs)

    def _load_dataset(self):
        archive = self.data_dir / "LJSpeech-1.1.tar.bz2"
        print("Скачивание LJSpeech...")
        wget.download(URL_LINK, str(archive))
        print("\nРаспаковка архива...")
        shutil.unpack_archive(str(archive), str(self.data_dir))
        extracted_dir = self.data_dir / "LJSpeech-1.1"
        for item in extracted_dir.iterdir():
            shutil.move(str(item), str(self.data_dir))
        shutil.rmtree(extracted_dir)
        archive.unlink()

    def _load_or_create_index(self):
        index_file = self.data_dir / "index.json"
        if index_file.exists():
            with open(index_file, 'r', encoding='utf-8') as f:
                index = json.load(f)
        else:
            index = self._create_index()
            with open(index_file, 'w', encoding='utf-8') as f:
                json.dump(index, f)
        return index

    def _create_index(self):
        index = []
        wavs_dir = self.data_dir / "wavs"
        if not wavs_dir.exists():
            self._load_dataset()
        wav_files = list(wavs_dir.glob("*.wav"))
        for wav_file in tqdm(wav_files, desc="Создание индекса..."):
            index.append({
                "path": str(wav_file.resolve())
            })
        return index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        item = self.index[idx]
        audio_path = item['path']

        waveform, sample_rate = torchaudio.load(audio_path)

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        audio_len = waveform.shape[-1]
        start = random.randint(0, max(0, audio_len - 8192))
        waveform = waveform[..., start: start + 8192]
        self.mel_spectrogram.mel_spectrogram.spectrogram.window = \
            self.mel_spectrogram.mel_spectrogram.spectrogram.window.to(waveform.device)
        mel_spec = self.mel_spectrogram(waveform).squeeze(0)

        return {
            'waveform': waveform,
            'mel_spec': mel_spec
        }
