from pathlib import Path

import pandas as pd

from src.logger.utils import plot_spectrogram
from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer
from itertools import chain
from src.datasets.melspectogram import MelSpectrogram, MelSpectrogramConfig
import torch
import itertools


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch)
        ref_waveform = batch["waveform"]
        ref_mel = batch["mel_spec"]

        with torch.no_grad():
            gen_waveform = self.model.generator(ref_mel)
        batch["wav_predict"] = gen_waveform

        mpd_real_out, mpd_real_feats = self.model.mpd(ref_waveform)
        mpd_fake_out, mpd_fake_feats = self.model.mpd(gen_waveform)
        msd_real_out, msd_real_feats = self.model.msd(ref_waveform)
        msd_fake_out, msd_fake_feats = self.model.msd(gen_waveform)

        disc_loss_dict = self.criterion.disc_loss_func(
            real_msd_logits=msd_real_out, fake_msd_logits=msd_fake_out,
            real_mpd_logits=mpd_real_out, fake_mpd_logits=mpd_fake_out
        )

        self.optimizer_disc.zero_grad()
        disc_loss_dict["disc_loss"].backward()
        self._clip_grad_norm_block(self.model.mpd)
        self._clip_grad_norm_block(self.model.msd)
        self.optimizer_disc.step()
        batch.update(disc_loss_dict)

        gen_waveform = self.model.generator(ref_mel)
        batch["wav_predict"] = gen_waveform

        mpd_real_out, mpd_real_feats = self.model.mpd(ref_waveform)
        mpd_fake_out, mpd_fake_feats = self.model.mpd(gen_waveform)
        msd_real_out, msd_real_feats = self.model.msd(ref_waveform)
        msd_fake_out, msd_fake_feats = self.model.msd(gen_waveform)

        melspec = MelSpectrogram(MelSpectrogramConfig())
        predict_mel = melspec(gen_waveform).squeeze(1)

        gen_loss_dict = self.criterion.gen_loss_func(
            fake_mels=predict_mel,
            real_mels=ref_mel,
            fake_msd_logits=msd_fake_out,
            fake_mpd_logits=mpd_fake_out,
            real_msd_activations=msd_real_feats,
            fake_msd_activations=msd_fake_feats,
            real_mpd_activations=mpd_real_feats,
            fake_mpd_activations=mpd_fake_feats
        )

        self.optimizer_gen.zero_grad()
        gen_loss_dict["gen_loss"].backward()
        self._clip_grad_norm_block(self.model.generator)
        self.optimizer_gen.step()

        batch.update(gen_loss_dict)

        for key_name in self.config.writer.loss_names:
            val = batch.get(key_name)
            if val is not None and torch.is_tensor(val):
                metrics.update(key_name, val.item())

        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            pred_wav = batch["wav_predict"][0].cpu()
            self.writer.add_audio("predict_wav", pred_wav, sample_rate=22050)
            wav = batch["waveform"][0].cpu()
            self.writer.add_audio("waveform", wav, sample_rate=22050)
            # Log Stuff
