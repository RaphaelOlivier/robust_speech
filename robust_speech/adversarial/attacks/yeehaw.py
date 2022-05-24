"""
Yeehaw Junction Attack (https://arxiv.org/abs/2203.05408) and Kenansville Attack (https://arxiv.org/abs/1910.05262)
In development
"""

import torch

from robust_speech.adversarial.attacks.attacker import Attacker


class YeehawJunctionAttack(Attacker):
    """
    Implementation of the Yeehaw Junction Attack
    (https://arxiv.org/abs/2203.05408).
    This attack is gradient independant and only considers the input audio
    and its Power Spectral Density.
    The implementation is based on the Armory implementation of the Kenansville attack
    (https://github.com/twosixlabs/armory/blob/master/armory/art_experimental/attacks/kenansville_dft.py)
    It differs from the original in that it is strictly model independant and the amount of noise is fixed as a hyperparameter.
    The original attack would search for the noise threshold that breaks the model.

    Arguments
    ---------
     asr_brain : rs.adversarial.brain.ASRBrain
        the brain object to attack
     targeted: bool
        if the attack is targeted (always true for now).
     snr: float
        SNR bound applied to the perturbation.
     snr_decimation: float
        SNR bound applied to the perturbation at the decimation step only. It should be higher than snr
     train_mode_for_backward: bool
        whether to force training mode in backward passes
        (necessary for RNN models)

    """

    def __init__(
        self, asr_brain, targeted=False, snr=100, snr_decimation=None,
        train_mode_for_backward=False
    ):
        """Carlini Wagner L2 Attack implementation in pytorch."""
        self.asr_brain = asr_brain
        # The last iteration (if we run many steps) repeat the search once.
        assert not targeted, "Kenansville is an untargeted attack"
        self.snr = snr
        self.snr_decimation = snr_decimation
        if snr_decimation is None:
            self.snr_decimation = self.snr + 20
        self.threshold_decimation = 10 ** (-self.snr_decimation / 10)
        self.threshold_clipping = 10 ** (-self.snr / 10) - \
            self.threshold_decimation
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

            # Decimation step
            wav_psd_index = torch.argsort(wav_psd)
            reordered = wav_psd[wav_psd_index]
            cumulative = torch.cumsum(reordered, dim=0)

            norm_threshold = self.threshold_decimation * cumulative[-1]
            j_dec = torch.searchsorted(cumulative, norm_threshold, right=True)
            wav_rfft[wav_psd_index[:j_dec]] = 0

            # Clipping step
            norm_threshold_clip = self.threshold_clipping * cumulative[-1]
            nfft = len(wav_rfft)

            diffs = torch.flip(reordered[1:]-reordered[:-1], dims=(0,))
            cumdiffs = diffs * \
                (torch.arange(
                    len(diffs), dtype=diffs.dtype, device=diffs.device))
            cumulative_clip = torch.cumsum(cumdiffs, dim=0)
            j_clip = nfft - torch.searchsorted(
                cumulative_clip, norm_threshold_clip, right=False)

            clip_threshold = reordered[j_clip-1]
            # + (cumclip - norm_threshold)/(nfft-j_clip)
            #print(norm_threshold, clip_threshold, reordered[-5:])
            clip_threshold_abs = torch.sqrt(clip_threshold/2)
            # print(torch.abs(
            #    wav_rfft[wav_psd_index[j_clip:]]))
            wav_rfft[wav_psd_index[j_clip:]] = wav_rfft[wav_psd_index[j_clip:]
                                                        ] / torch.abs(wav_rfft[wav_psd_index[j_clip:]]) * clip_threshold_abs

            wav_psd = torch.abs(wav_rfft) ** 2
            if len(wav) % 2:  # odd: DC frequency
                wav_psd[1:] *= 2
            else:  # even: DC and Nyquist frequencies
                wav_psd[1:-1] *= 2
            # print(torch.abs(
            #    wav_rfft[wav_psd_index[j_clip:]]))
            # print(torch.abs(
            #    wav_rfft[wav_psd_index[j_clip:]])**2 * 2)
            # Zero out low power frequencies and invert to time domain
            wav = torch.fft.irfft(wav_rfft, len(wav)).type(wav.dtype)

            wavs[i, :len_wav] = wav
        return wavs


class KenansvilleAttack(YeehawJunctionAttack):
    """
    Implementation of the Kenansville Attack
    (https://arxiv.org/abs/1910.05262).
    Similar to Yeehaw without the clipping step

    Arguments
    ---------
     asr_brain : rs.adversarial.brain.ASRBrain
        the brain object to attack
     targeted: bool
        if the attack is targeted (always true for now).
     snr: float
        SNR bound applied to the perturbation.
     train_mode_for_backward: bool
        whether to force training mode in backward passes
        (necessary for RNN models)
    """

    def __init__(
        self, asr_brain, targeted=False, snr=100,
        train_mode_for_backward=False
    ):

        super(KenansvilleAttack, self).__init__(
            asr_brain,
            targeted=targeted,
            snr=snr,
            snr_decimation=snr,
            train_mode_for_backward=train_mode_for_backward
        )
