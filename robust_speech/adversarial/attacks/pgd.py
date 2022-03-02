
import numpy as np
import torch
import torch.nn as nn

import speechbrain as sb 

from advertorch.attacks.iterative_projected_gradient import L2PGDAttack
from advertorch.attacks.base import Attack,LabelMixin
from advertorch.utils import clamp
from advertorch.utils import normalize_by_pnorm
from advertorch.utils import clamp_by_pnorm
from advertorch.utils import is_float_or_torch_tensor
from advertorch.utils import batch_multiply
from advertorch.utils import batch_clamp
from advertorch.utils import replicate_input
from advertorch.utils import batch_l1_proj
from advertorch.attacks.utils import rand_init_delta

import robust_speech as rs

def reverse_bound_from_rel_bound(batch,rel):
    wavs, wav_lens = batch.sig
    wav_lens = [int(wavs.size(1)*r) for r in wav_lens]
    epss = []
    for i in range(len(wavs)):
        eps = torch.norm(wavs[i,:wav_lens[i]])/rel
        epss.append(eps)
    return torch.tensor(epss).to(wavs.device)


def perturb_iterative(batch, asr_brain, nb_iter, eps, eps_iter,
                      delta_init=None, minimize=False, ord=np.inf,
                      clip_min=None, clip_max=None,
                      l1_sparsity=None):
    """
    Iteratively maximize the loss over the input. It is a shared method for
    iterative attacks including IterativeGradientSign, LinfPGD, etc.
    :param xvar: input data.
    :param yvar: input labels.
    :param predict: forward pass function.
    :param nb_iter: number of iterations.
    :param eps: maximum distortion.
    :param eps_iter: attack step size.
    :param loss_fn: loss function.
    :param delta_init: (optional) tensor contains the random initialization.
    :param minimize: (optional bool) whether to minimize or maximize the loss.
    :param ord: (optional) the order of maximum distortion (inf or 2).
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :param l1_sparsity: sparsity value for L1 projection.
                  - if None, then perform regular L1 projection.
                  - if float value, then perform sparse L1 descent from
                    Algorithm 1 in https://arxiv.org/pdf/1904.13000v1.pdf
    :return: tensor containing the perturbed input.
    """
    wav_init, wav_lens = batch.sig
    if delta_init is not None:
        delta = delta_init
    else:
        delta = torch.zeros_like(wav_init) 
        
    delta.requires_grad_()
    for ii in range(nb_iter):
        batch.sig = wav_init+delta, wav_lens
        predictions = asr_brain.compute_forward(batch, rs.Stage.ATTACK)
        loss = asr_brain.compute_objectives(predictions,batch,rs.Stage.ATTACK)
        if minimize:
            loss = -loss
        loss.backward()
        if ord == np.inf:
            grad_sign = delta.grad.data.sign()
            delta.data = delta.data + batch_multiply(eps_iter, grad_sign)
            delta.data = batch_clamp(eps, delta.data)
            delta.data = clamp(wav_init.data + delta.data, clip_min, clip_max
                               ) - wav_init.data

        elif ord == 2:
            grad = delta.grad.data
            grad = normalize_by_pnorm(grad)
            delta.data = delta.data + batch_multiply(eps_iter, grad)
            delta.data = clamp(wav_init.data + delta.data, clip_min, clip_max
                               ) - wav_init.data
            if eps is not None:
                delta.data = clamp_by_pnorm(delta.data, ord, eps)
        elif ord == 1:
            grad = delta.grad.data
            abs_grad = torch.abs(grad)

            batch_size = grad.size(0)
            view = abs_grad.view(batch_size, -1)
            view_size = view.size(1)
            if l1_sparsity is None:
                vals, idx = view.topk(1)
            else:
                vals, idx = view.topk(
                    int(np.round((1 - l1_sparsity) * view_size)))

            out = torch.zeros_like(view).scatter_(1, idx, vals)
            out = out.view_as(grad)
            grad = grad.sign() * (out > 0).float()
            grad = normalize_by_pnorm(grad, p=1)
            delta.data = delta.data + batch_multiply(eps_iter, grad)

            delta.data = batch_l1_proj(delta.data.cpu(), eps)
            delta.data = delta.data.to(wav_init.device)
            delta.data = clamp(wav_init.data + delta.data, clip_min, clip_max
                               ) - wav_init.data
        else:
            error = "Only ord = inf, ord = 1 and ord = 2 have been implemented"
            raise NotImplementedError(error)
        delta.grad.data.zero_()

    wav_adv = clamp(wav_init + delta, clip_min, clip_max)
    return wav_adv


class ASRPGDAttack(Attack, LabelMixin):
    """
    The projected gradient descent attack (Madry et al, 2017).
    The attack performs nb_iter steps of size eps_iter, while always staying
    within eps from the initial point.
    Paper: https://arxiv.org/pdf/1706.06083.pdf
    :param predict: forward pass function.
    :param loss_fn: loss function.
    :param eps: maximum distortion.
    :param nb_iter: number of iterations.
    :param eps_iter: attack step size.
    :param rand_init: (optional bool) random initialization.
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :param ord: (optional) the order of maximum distortion (inf or 2).
    :param targeted: if the attack is targeted.
    """

    def __init__(
            self, asr_brain, eps=0.3, nb_iter=40,
            rel_eps_iter=0.1, rand_init=True, clip_min=None, clip_max=None,
            ord=np.inf, l1_sparsity=None, targeted=False, train_mode_for_backward=True):
        """
        Create an instance of the PGDAttack.
        """
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.eps = eps
        self.nb_iter = nb_iter
        self.rel_eps_iter = rel_eps_iter
        self.rand_init = rand_init
        self.ord = ord
        self.targeted = targeted
        self.asr_brain=asr_brain
        self.l1_sparsity = l1_sparsity
        self.train_mode_for_backward=train_mode_for_backward
        assert is_float_or_torch_tensor(self.rel_eps_iter)
        assert is_float_or_torch_tensor(self.eps)

    def perturb(self, batch):
        if self.train_mode_for_backward:
            self.asr_brain.module_train()
        else:
            self.asr_brain.module_eval()
        """
        Given examples (x, y), returns their adversarial counterparts with
        an attack length of eps.
        :param x: input tensor.
        :param y: label tensor.
                  - if None and self.targeted=False, compute y as predicted
                    labels.
                  - if self.targeted=True, then y must be the targeted labels.
        :return: tensor containing perturbed inputs.
        """
        
        save_device = batch.sig[0].device
        batch = batch.to(self.asr_brain.device)
        save_input = batch.sig[0]
        x = torch.clone(save_input)
        delta = torch.zeros_like(x)
        delta = nn.Parameter(delta)
        if self.rand_init:
            clip_min = self.clip_min if self.clip_min is not None else -0.1
            clip_max = self.clip_max if self.clip_max is not None else 0.1
            rand_init_delta(
                delta, x, self.ord, self.eps, clip_min, clip_max)
            delta.data = clamp(
                x + delta.data, min=self.clip_min, max=self.clip_max) - x

        wav_adv = perturb_iterative(
            batch, self.asr_brain, nb_iter=self.nb_iter,
            eps=self.eps, eps_iter=self.rel_eps_iter*self.eps,
            minimize=self.targeted, ord=self.ord, 
            clip_min=self.clip_min, clip_max=self.clip_max, 
            delta_init=delta, l1_sparsity=self.l1_sparsity
        )

        batch.sig = save_input, batch.sig[1]
        batch = batch.to(save_device)
        return wav_adv.data.to(save_device)



class ASRL2PGDAttack(ASRPGDAttack):
    """
    PGD Attack with order=L2
    :param predict: forward pass function.
    :param loss_fn: loss function.
    :param eps: maximum distortion.
    :param nb_iter: number of iterations.
    :param eps_iter: attack step size.
    :param rand_init: (optional bool) random initialization.
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :param targeted: if the attack is targeted.
    """

    def __init__(
            self, asr_brain, eps=0.3, nb_iter=40,
            rel_eps_iter=0.1, rand_init=True, clip_min=None, clip_max=None,
            targeted=False, train_mode_for_backward=True):
        ord = 2
        super(ASRL2PGDAttack, self).__init__(
            asr_brain=asr_brain, eps=eps, nb_iter=nb_iter,
            rel_eps_iter=rel_eps_iter, rand_init=rand_init, clip_min=clip_min,
            clip_max=clip_max, targeted=targeted, train_mode_for_backward=train_mode_for_backward,
            ord=ord)


class ASRLinfPGDAttack(ASRPGDAttack):
    """
    PGD Attack with order=Linf
    :param predict: forward pass function.
    :param loss_fn: loss function.
    :param eps: maximum distortion.
    :param nb_iter: number of iterations.
    :param rel_eps_iter: relative attack step size.
    :param rand_init: (optional bool) random initialization.
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :param targeted: if the attack is targeted.
    """

    def __init__(
            self, asr_brain, eps=0.001, nb_iter=40,
            rel_eps_iter=0.1, rand_init=True, clip_min=None, clip_max=None,
            targeted=False, train_mode_for_backward=True):
        ord = np.inf
        super(ASRLinfPGDAttack, self).__init__(
            asr_brain, eps=eps, nb_iter=nb_iter,
            rel_eps_iter=rel_eps_iter, rand_init=rand_init, clip_min=clip_min,
            clip_max=clip_max, targeted=targeted, train_mode_for_backward=train_mode_for_backward,
            ord=ord)


class SNRPGDAttack(ASRL2PGDAttack):
    def __init__(
            self, asr_brain, snr=40, nb_iter=40,
            rel_eps_iter=0.1, rand_init=True, clip_min=None, clip_max=None,
            targeted=False, train_mode_for_backward=True):
        super(SNRPGDAttack, self).__init__(
            asr_brain=asr_brain, eps=1.0, nb_iter=nb_iter,
            rel_eps_iter=rel_eps_iter, rand_init=rand_init, clip_min=clip_min,
            clip_max=clip_max, targeted=targeted, train_mode_for_backward=train_mode_for_backward)
        assert isinstance(snr,int)
        self.rel_eps=torch.pow(torch.tensor(10.),float(snr)/20)

    def perturb(self, batch):
        """
        Given examples (x, y), returns their adversarial counterparts with
        an attack length of eps.
        :param x: input tensor.
        :param y: label tensor.
                  - if None and self.targeted=False, compute y as predicted
                    labels.
                  - if self.targeted=True, then y must be the targeted labels.
        :return: tensor containing perturbed inputs.
        """

        save_device = batch.sig[0].device
        batch = batch.to(self.asr_brain.device)
        self.eps = reverse_bound_from_rel_bound(batch,self.rel_eps)
        res = super(SNRPGDAttack, self).perturb(batch)
        self.eps=1.0
        batch.to(save_device)
        return res.to(save_device)

