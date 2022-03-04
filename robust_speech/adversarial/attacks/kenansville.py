
import torch
import torch.nn as nn
import torch.optim as optim

#from advertorch.utils import calc_l2distsq
from advertorch.utils import replicate_input
from advertorch.attacks.base import Attack
from advertorch.attacks.base import LabelMixin
from robust_speech.adversarial.attacks.attacker import Attacker

import speechbrain as sb

import robust_speech as rs

def is_successful(y1, y2, targeted):
    equal = type(y1) == type(y2)
    if equal:
        if isinstance(y1,torch.Tensor):
            equal = y1.size()==y2.size() and (y1==y2).all()
        else:
            equal = y1 == y2
    return targeted == equal

def calc_l2distsq(x, y, mask):
    d = ((x - y)**2) * mask
    return d.view(d.shape[0], -1).sum(dim=1) #/ mask.view(mask.shape[0], -1).sum(dim=1)


class KenansvilleAttack(Attacker):
    """
    The Kenansville Attack, https://arxiv.org/abs/1910.05262
    """

    def __init__(self, asr_brain, targeted=False,snr_db=100,train_mode_for_backward=False):
        """Carlini Wagner L2 Attack implementation in pytorch."""
        self.asr_brain = asr_brain
        # The last iteration (if we run many steps) repeat the search once.
        assert not targeted, "Kenansville is an untargeted attack"
        self.snr_db = snr_db
        self.threshold = 10 ** (-self.snr_db / 10)
        self.train_mode_for_backward=train_mode_for_backward # not used, for compatibility only

    def perturb(self, batch):
        #save_device = batch.sig[0].device
        #batch = batch.to(self.asr_brain.device)
        wavs, rel_lengths = batch.sig 
        wavs = replicate_input(wavs)
        batch_size = wavs.size(0)
        wav_lengths = (rel_lengths.float()*wavs.size(1)).long()

        for i in range(batch_size):
            x,n = wavs[i],wav_lengths[i] 
            x_rfft = torch.fft.rfft(x)
            x_psd = torch.abs(x_rfft) ** 2
            if len(x) % 2:  # odd: DC frequency
                x_psd[1:] *= 2
            else:  # even: DC and Nyquist frequencies
                x_psd[1:-1] *= 2

            # Scale the threshold based on the power of the signal
            # Find frequencies in order with cumulative perturbation less than threshold
            #     Sort frequencies by power density in ascending order
            x_psd_index = torch.argsort(x_psd)
            reordered = x_psd[x_psd_index]
            cumulative = torch.cumsum(reordered,dim=0)
            norm_threshold = self.threshold * cumulative[-1]
            j = torch.searchsorted(cumulative, norm_threshold, right=True)

            # Zero out low power frequencies and invert to time domain
            x_rfft[x_psd_index[:j]] = 0
            x = torch.fft.irfft(x_rfft, len(x)).type(x.dtype)
            wavs[i,:n]=x


        return wavs