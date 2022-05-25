import numpy as np
import logging
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd import gradcheck
import speechbrain as sb

logger = logging.getLogger(__name__)


class SpeechNoiseAugmentation(nn.Module):
    def __init__(self, sigma, filter=None, enhancer=None):
        self.sigma = sigma
        if filter is not None:
            self.filter = filter(sigma=sigma)
        else:
            self.filter = None
        self.enhancer = enhancer

    def forward(self, sigs, sig_lens):
        """
        Augment the sample `batch` with Gaussian noise.
        """
        sigs_enh = []
        sig_lens = [int(l*sigs.size(1)) for l in sig_lens]
        for idx, sig in enumerate(sigs):
            sig_len = sig_lens[idx]
            x_enh = SmoothCh.apply(
                sig, int(sig_len), self.sigma)

            sigs_enh.append(x_enh)

        sigs_enh = torch.stack(sigs_enh)
        if self.filter is not None:
            sigs_enh = self.filter(sigs_enh)

        if self.enhancer is not None:
            sigs_enh = self.enhancer.enhance_batch(sigs_enh, lengths=sig_lens)
        return sigs_enh

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


def augment(x: np.ndarray, sigma) -> np.ndarray:
    x_aug = np.copy(x)
    noise = np.random.normal(0, scale=sigma, size=x.shape)
    x_aug = (x+noise)
    return x_aug


def smooth_np(x, sigma):
    x_aug = np.copy(x)
    noise = np.random.normal(0, scale=sigma, size=x.shape)
    x_aug = (x+noise)
    return x_aug


class SmoothCh(Function):
    @staticmethod
    def forward(ctx, x, x_len, sigma):
        x_ = x.clone()
        x_np = x_.detach().cpu().numpy()
        x_np[:x_len] = smooth_np(x_np[:x_len], sigma)
        x_enh = torch.tensor(x_np).to(x_.device)
        return x_enh

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None, None
