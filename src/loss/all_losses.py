import torch
import torch.nn as nn
import torch.nn.functional as F


class HiFiGANLoss(nn.Module):
    def __init__(self, fm_lambda, mel_lambda):
        super().__init__()
        self.fm_lambda = fm_lambda
        self.mel_lambda = mel_lambda

    def disc_adv_loss(self, real_outputs, fake_outputs):
        device = real_outputs[0].device
        res = torch.zeros(1, device=device)
        for r, f in zip(real_outputs, fake_outputs):
            res = res + torch.mean((r - 1) ** 2) + torch.mean(f ** 2)
        return res

    def gen_adv_loss(self, fake_outputs):
        device = fake_outputs[0].device
        res = torch.zeros(1, device=device)
        for f in fake_outputs:
            res = res + torch.mean((f - 1) ** 2)
        return res

    def mel_loss(self, real_mel, fake_mel):
        return self.mel_lambda * F.l1_loss(fake_mel, real_mel)

    def fm_loss(self, real_feats, fake_feats):
        device = real_feats[0][0].device
        res = torch.zeros(1, device=device)
        for rf, ff in zip(real_feats, fake_feats):
            for r, f in zip(rf, ff):
                res = res + F.l1_loss(f, r)
        return res * self.fm_lambda

    def disc_loss_func(
        self,
        real_msd_logits, fake_msd_logits,
        real_mpd_logits, fake_mpd_logits
    ):
        msd_disc_loss = self.disc_adv_loss(real_msd_logits, fake_msd_logits)
        mpd_disc_loss = self.disc_adv_loss(real_mpd_logits, fake_mpd_logits)
        D_loss = msd_disc_loss + mpd_disc_loss
        return {"disc_loss": D_loss}

    def gen_loss_func(
        self,
        fake_mels,
        real_mels,
        fake_msd_logits,
        fake_mpd_logits,
        real_msd_activations,
        fake_msd_activations,
        real_mpd_activations,
        fake_mpd_activations
    ):
        msd_gen_loss = self.gen_adv_loss(fake_msd_logits)
        mpd_gen_loss = self.gen_adv_loss(fake_mpd_logits)

        adv_gan_loss = msd_gen_loss + mpd_gen_loss
        fm_loss_val = self.fm_loss(real_msd_activations, fake_msd_activations) + \
            self.fm_loss(real_mpd_activations, fake_mpd_activations)
        mel_loss_val = self.mel_loss(real_mels, fake_mels)

        gan_loss = adv_gan_loss + fm_loss_val + mel_loss_val

        return {
            "mel_loss": mel_loss_val,
            "fm_loss": fm_loss_val,
            "adv_loss": adv_gan_loss,
            "gen_loss": gan_loss
        }
