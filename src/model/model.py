import torch.nn as nn
from src.model.generator import Generator
from src.model.mpd import MPD
from src.model.msd import MSD


class HiFiGAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.generator = Generator()
        self.mpd = MPD()
        self.msd = MSD()

    def forward_generator(self, mel_spectrogram):
        return self.generator(mel_spectrogram)

    def forward_discriminators(self, waveforms):
        mpd_out, mpd_feats = self.mpd(waveforms)
        msd_out, msd_feats = self.msd(waveforms)
        return mpd_out, msd_out, mpd_feats, msd_feats

    def forward(self, mel_spectrogram):
        waveforms = self.forward_generator(mel_spectrogram)
        mpd_out, msd_out, mpd_feats, msd_feats = self.forward_discriminators(waveforms)
        return waveforms, mpd_out, msd_out, mpd_feats, msd_feats

    def __str__(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (
            f"All params: {total_params}\n"
            f"Trainable params: {trainable_params}\n"
        )
