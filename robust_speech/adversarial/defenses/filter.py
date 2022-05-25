from audlib.enhance import wiener_iter, asnr, SSFEnhancer
from audlib.sig.window import hamming
import numpy as np
import torch
from torch.autograd import Function


class ASNRWiener:
    def __init__(self, sr, nfft, hop, sigma=None):
        super(ASNRWiener, self).__init__()
        self.sr = sr
        self.nfft = nfft
        self.hop = hop
        self.window = hamming(self.nfft, hop=self.hop)
        self.gaussian_sigma = sigma
        self.lpc_order = 12

    def __call__(self, x):
        return ASNRWienerCh.apply(x, self.sr, self.nfft, self.hop, self.window, self.gaussian_sigma, self.lpc_order)


class ASNRWienerCh(Function):
    @staticmethod
    def forward(ctx, x, sr, nfft, hop, window, gaussian_sigma, lpc_order):
        x_np = x.cpu().detach().numpy()
        for i in range(len(x_np)):
            noise = np.random.normal(
                0, scale=gaussian_sigma, size=x_np[i].shape) if gaussian_sigma else None
            filtered_output, _ = asnr(x_np[i], sr, window, hop, nfft, noise=(
                noise if gaussian_sigma is not None else None), snrsmooth=0.98, noisesmooth=0.98, llkthres=.15, zphase=True, rule="wiener")
            if len(filtered_output) < len(x_np[i]):
                filtered_output = np.pad(filtered_output, mode="mean", pad_width=(
                    (0, len(x_np[i])-len(filtered_output))))
            elif len(filtered_output) > len(x_np[i]):
                filtered_output = filtered_output[:len(x_np[i])]
            if not np.isnan(filtered_output).any():
                x[i] = torch.from_numpy(filtered_output)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None, None, None, None, None, None, None
