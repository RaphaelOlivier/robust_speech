"""
Metrics and loggers for adversarial attacks.
"""

import os

import torch
import torchaudio
from speechbrain.utils.edit_distance import accumulatable_wer_stats
from speechbrain.utils.metric_stats import MetricStats


def snr(audio, perturbation, rel_length=torch.tensor([1.0])):
    """
    Signal to Noise Ratio computation

    Arguments
    ---------
    audio : torch.tensor
        the original padded audio
    perturbation : torch.tensor
        the padded perturbation
    rel_length : torch.tensor
        the relative length of the wavs in the batch
    """

    length = (audio.size(1) * rel_length).long()
    num = torch.tensor(
        [torch.square(audio[i, : length[i]]).sum()
         for i in range(audio.size(0))]
    )
    den = torch.tensor(
        [torch.square(perturbation[i, : length[i]]).sum()
         for i in range(audio.size(0))]
    )
    ratio = 10 * torch.log10(num / den)
    return torch.round(ratio).long()


class SNRComputer(MetricStats):
    """Tracks Signal to Noise Ratio"""

    def __init__(self, **kwargs):
        def metric(batch, adv_wav):
            return snr(batch.sig[0], adv_wav - batch.sig[0], batch.sig[1])

        super().__init__(metric, **kwargs)


class AudioSaver:
    """
    Class used to save audio files computed with adversarial attacks.

    Arguments
    ---------
    save_audio_path: optional string
        path to the folder in which to save audio files
    sample_rate: int
        audio sample rate for wav encoding
    """

    def __init__(self, save_audio_path, sample_rate=16000):
        self.save_audio_path = save_audio_path
        self.sample_rate = sample_rate
        if os.path.exists(self.save_audio_path):
            if not os.path.isdir(self.save_audio_path):
                raise ValueError("%f not a directory" % self.save_audio_path)
        else:
            os.makedirs(save_audio_path)

    def save(self, audio_ids, batch, adv_sig):
        """Save a batch of audio files, both natural and adversarial"""
        bs = len(audio_ids)
        lengths = (batch.sig[0].size(1) * batch.sig[1]).long()
        for i in range(bs):
            id = audio_ids[i]
            wav = batch.sig[0][i, : lengths[i]].detach().cpu().unsqueeze(0)
            adv_wav = adv_sig[i, : lengths[i]].detach().cpu().unsqueeze(0)
            self.save_wav(id, wav, adv_wav)

    def save_wav(self, id, wav, adv_wav):
        """Save the original and the adversarial versions of a single audio file"""
        print(wav.size())
        nat_path = id + "_nat.wav"
        adv_path = id + "_adv.wav"
        torchaudio.save(
            os.path.join(self.save_audio_path, nat_path), wav, self.sample_rate
        )
        torchaudio.save(
            os.path.join(self.save_audio_path,
                         adv_path), adv_wav, self.sample_rate
        )
