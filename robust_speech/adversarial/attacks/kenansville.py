"""
Kenansville Attack (https://arxiv.org/abs/1910.05262)
"""

import torch

from robust_speech.adversarial.attacks.attacker import Attacker


class KenansvilleAttack(Attacker):
    """
    Implementation of the Kenansville Attack (https://arxiv.org/abs/1910.05262).
    This attack is model independant and only considers the input audio and its Power Spectral Density.
    The implementation is based on the Armory one (https://github.com/twosixlabs/armory/blob/master/armory/art_experimental/attacks/kenansville_dft.py)

    Arguments
    ---------
     asr_brain : rs.adversarial.brain.ASRBrain
        the brain object to attack
     targeted: bool
        if the attack is targeted (always true for now).
     snr: float
        Linf bound applied to the perturbation.
     train_mode_for_backward: bool
        whether to force training mode in backward passes (necessary for RNN models)

    """

    def __init__(
        self, asr_brain, targeted=False, snr=100, train_mode_for_backward=False
    ):
        """Carlini Wagner L2 Attack implementation in pytorch."""
        self.asr_brain = asr_brain
        # The last iteration (if we run many steps) repeat the search once.
        assert not targeted, "Kenansville is an untargeted attack"
        self.snr = snr
        self.threshold = 10 ** (-self.snr / 10)
        # not used, for compatibility only
        self.train_mode_for_backward = train_mode_for_backward

    def perturb(self, batch):
        """
        Compute an adversarial perturbation

        Arguments
        ---------
        batch : sb.PaddedBatch
            The input batch to perturb

        Returns
        -------
        the tensor of the perturbed batch
        """
        wavs, rel_lengths = batch.sig
        wavs = wavs.detach().clone()
        batch_size = wavs.size(0)
        wav_lengths = (rel_lengths.float() * wavs.size(1)).long()

        for i in range(batch_size):
            wav, len_wav = wavs[i, : wav_lengths[i]], wav_lengths[i]
            wav_rfft = torch.fft.rfft(wav)
            wav_psd = torch.abs(wav_rfft) ** 2
            if len(wav) % 2:  # odd: DC frequency
                wav_psd[1:] *= 2
            else:  # even: DC and Nyquist frequencies
                wav_psd[1:-1] *= 2

            # Scale the threshold based on the power of the signal
            # Find frequencies in order with cumulative perturbation less than threshold
            #     Sort frequencies by power density in ascending order
            wav_psd_index = torch.argsort(wav_psd)
            reordered = wav_psd[wav_psd_index]
            cumulative = torch.cumsum(reordered, dim=0)
            norm_threshold = self.threshold * cumulative[-1]
            j = torch.searchsorted(cumulative, norm_threshold, right=True)

            # Zero out low power frequencies and invert to time domain
            wav_rfft[wav_psd_index[:j]] = 0
            wav = torch.fft.irfft(wav_rfft, len(wav)).type(wav.dtype)
            wavs[i, :len_wav] = wav
        return wavs
