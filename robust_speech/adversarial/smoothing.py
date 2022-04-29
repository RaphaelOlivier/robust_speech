from art.defences.preprocessor.gaussian_augmentation import GaussianAugmentation
from art.config import ART_NUMPY_DTYPE
import numpy as np
import logging
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd import gradcheck
import speechbrain as sb

logger = logging.getLogger(__name__)

class SpeechNoiseAugmentation(GaussianAugmentation,nn.Module):
    def __init__(self, *args, filter=None, high_freq=False, **kwargs):
        nn.Module.__init__(self)
        GaussianAugmentation.__init__(self, *args, **kwargs)
        self.filter = filter
        self.high_freq=high_freq
 
    def forward(self, batch):
        """
        Augment the sample `batch` with Gaussian noise.
        """
        sigs, sig_lens = batch.sig
        tokens, token_lens = batch.tokens

        sigs_enh = []
        sigs_enh_lens = []
        tokens_enh = []
        tokens_enh_lens = []
        for idx, sig in enumerate(sigs):
            y = tokens[idx]
            sig_len = sig_lens[idx]
            x_enh, y_enh = SmoothCh.apply(sig, int(sig_len), y, self.sigma, self.high_freq, self.filter)
            y_enh_len = token_lens[idx]

            sigs_enh.append(x_enh)
            sigs_enh_lens.append(sig_len)
            tokens_enh.append(y_enh)
            tokens_enh_lens.append(y_enh_len)
        
        sigs_enh = torch.stack(sigs_enh)
        sigs_enh_lens = torch.stack(sigs_enh_lens)
        tokens_enh = torch.stack(tokens_enh)
        tokens_enh_lens = torch.stack(tokens_enh_lens)

        batch.sig = (sigs_enh, sigs_enh_lens)
        batch.tokens = (tokens_enh, tokens_enh_lens)
                
        return batch

    def __call__(self,*args,**kwargs):
        return self.forward(*args,**kwargs)


def augment(x: np.ndarray,sigma,high_freq) -> np.ndarray:
    x_aug = np.copy(x)
    if high_freq:
        noise = np.random.normal(0, scale=sigma, size=(x.shape[0]+1,))
        noise = 0.5 * (noise[1:]-noise[:-1])
    else:
        noise = np.random.normal(0, scale=sigma, size=x.shape)
    x_aug = (x+noise).astype(ART_NUMPY_DTYPE)
    return x_aug


def smooth_np(x, y, sigma,high_freq, filter=None):
    x_aug = np.copy(x)
    if high_freq:
        noise = np.random.normal(0, scale=sigma, size=(x.shape[0]+1,))
        noise = 0.5 * (noise[1:]-noise[:-1])
    else:
        noise = np.random.normal(0, scale=sigma, size=x.shape)
    x_aug = (x+noise).astype(ART_NUMPY_DTYPE)
    if filter is not None:
        x_aug = filter(x_aug)
    return x_aug, y


class SmoothCh(Function):
    @staticmethod
    def forward(ctx, x, x_len, y, sigma, high_freq, filter=None):
        x_=x.clone()
        y_= y.clone()
        x_np=x_.detach().cpu().numpy()
        y_np = y_.detach().cpu().numpy()
        x_np[:x_len], y_enh = smooth_np(x_np[:x_len], y_np, sigma, high_freq, filter)
        x_enh = torch.tensor(x_np).to(x_.device)
        y_enh = torch.tensor(y_enh).to(y_.device)
        return x_enh, y_enh

    @staticmethod
    def backward(ctx,grad_output):
        grad_input=grad_output.clone()
        return grad_input, None, None, None
