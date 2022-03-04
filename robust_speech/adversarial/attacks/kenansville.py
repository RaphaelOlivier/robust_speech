

import torch

#from advertorch.utils import calc_l2distsq
from advertorch.utils import replicate_input
from advertorch.attacks.base import Attack
from advertorch.attacks.base import LabelMixin
from robust_speech.adversarial.attacks.attacker import Attacker

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
    Implementation of the Kenansville Attack (https://arxiv.org/abs/1910.05262).
    This attack is model independant and only considers the input audio and its Power Spectral Density.
    The implementation is based on the Armory one (https://github.com/twosixlabs/armory/blob/master/armory/art_experimental/attacks/kenansville_dft.py)
    
    Arguments
    ---------
     asr_brain : rs.adversarial.brain.ASRBrain
        the brain object to attack
     targeted: bool
        if the attack is targeted (always true for now).
     snr_db: float
        Linf bound applied to the perturbation.
     train_mode_for_backward: bool
        whether to force training mode in backward passes (necessary for RNN models)

    
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