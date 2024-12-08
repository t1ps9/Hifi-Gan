from torch.utils.data import Dataset
from pathlib import Path
import torchaudio
from src.datasets.melspectogram import MelSpectrogram, MelSpectrogramConfig
import random


class CustomWavDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.wav_files = list(self.data_dir.glob("*.wav"))
        self.transform = transform
        self.mel_spectrogram = MelSpectrogram(MelSpectrogramConfig())

    def __len__(self):
        return len(self.wav_files)

    def __getitem__(self, idx):
        wav_path = self.wav_files[idx]
        waveform, sr = torchaudio.load(wav_path)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        full_waveform = waveform.clone()
        audio_len = waveform.shape[-1]
        start = random.randint(0, max(0, audio_len - 8192))
        waveform = waveform[..., start: start + 8192]

        mel_spec = self.mel_spectrogram(waveform).squeeze(0)
        full_mel = self.mel_spectrogram(full_waveform).squeeze(0)

        return {
            'waveform': waveform,
            'mel_spec': mel_spec,
            'full_waveform': full_waveform,
            'full_mel': full_mel
        }
