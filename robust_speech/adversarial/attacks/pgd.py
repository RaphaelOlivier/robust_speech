import numpy as np
import torch
import torch.nn as nn

from advertorch.utils import clamp
from advertorch.utils import normalize_by_pnorm
from advertorch.utils import clamp_by_pnorm
from advertorch.utils import batch_multiply
from advertorch.utils import batch_clamp
from advertorch.utils import batch_l1_proj
from advertorch.attacks.utils import rand_init_delta

import robust_speech as rs

from robust_speech.adversarial.attacks.attacker import Attacker


def reverse_bound_from_rel_bound(batch, rel, ord=2):
   wavs, wav_lens = batch.sig
   wav_lens = [int(wavs.size(1)*r) for r in wav_lens]
   epss = []
   for i in range(len(wavs)):
      eps = torch.norm(wavs[i, :wav_lens[i]],p=ord)/rel
      epss.append(eps)
   return torch.tensor(epss).to(wavs.device)


def perturb_iterative(batch, asr_brain, nb_iter, eps, eps_iter,
                      delta_init=None, minimize=False, ord=np.inf,
                      clip_min=None, clip_max=None,
                      l1_sparsity=None):
   """
   Iteratively maximize the loss over the input.

   Arguments
   ---------
   asr_brain: rs.adversarial.brain.ASRBrain
      brain object.
   eps: float
      maximum distortion.
   nb_iter: int
      number of iterations.
   eps_iter: float
      attack step size.
   rand_init: (optional bool) 
      random initialization.
   clip_min: (optional) float
      mininum value per input dimension.
   clip_max: (optional) float
      maximum value per input dimension.
   ord: (optional) int
      the order of maximum distortion (inf or 2).
   targeted: bool
      if the attack is targeted.
   l1_sparsity: optional float
      sparsity value for L1 projection.
         - if None, then perform regular L1 projection.
         - if float value, then perform sparse L1 descent from
         Algorithm 1 in https://arxiv.org/pdf/1904.13000v1.pdf


   Returns
   -------
   tensor containing the perturbed input.
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
      loss = asr_brain.compute_objectives(
         predictions, batch, rs.Stage.ATTACK)
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
      # print(loss)
   wav_adv = clamp(wav_init + delta, clip_min, clip_max)
   return wav_adv


class ASRPGDAttack(Attacker):
   """
   Implementation of the PGD attack (https://arxiv.org/abs/1706.06083)
   Based on the Advertorch implementation (https://github.com/BorealisAI/advertorch/blob/master/advertorch/attacks/iterative_projected_gradient.py)
   The attack performs nb_iter steps of size eps_iter, while always staying
   within eps from the initial point.

   Arguments
   ---------
   asr_brain: rs.adversarial.brain.ASRBrain
      brain object.
   eps: float
      maximum distortion.
   nb_iter: int
      number of iterations.
   eps_iter: float
      attack step size.
   rand_init: (optional bool) 
      random initialization.
   clip_min: (optional) float
      mininum value per input dimension.
   clip_max: (optional) float
      maximum value per input dimension.
   ord: (optional) int
      the order of maximum distortion (inf or 2).
   targeted: bool
      if the attack is targeted.
   train_mode_for_backward: bool
      whether to force training mode in backward passes (necessary for RNN models)
   """

   def __init__(
         self, asr_brain, eps=0.3, nb_iter=40,
         rel_eps_iter=0.1, rand_init=True, clip_min=None, clip_max=None,
         ord=np.inf, l1_sparsity=None, targeted=False, train_mode_for_backward=True):

      self.clip_min = clip_min
      self.clip_max = clip_max
      self.eps = eps
      self.nb_iter = nb_iter
      self.rel_eps_iter = rel_eps_iter
      self.rand_init = rand_init
      self.ord = ord
      self.targeted = targeted
      self.asr_brain = asr_brain
      self.l1_sparsity = l1_sparsity
      self.train_mode_for_backward = train_mode_for_backward

      assert isinstance(self.rel_eps_iter, torch.Tensor) or isinstance(self.rel_eps_iter, float)
      assert isinstance(self.eps, torch.Tensor) or isinstance(self.eps, float)

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
      if self.train_mode_for_backward:
         self.asr_brain.module_train()
      else:
         self.asr_brain.module_eval()

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
      self.asr_brain.module_eval()
      return wav_adv.data.to(save_device)


class ASRL2PGDAttack(ASRPGDAttack):
   """
   PGD Attack with order=L2
   Arguments
   ---------
   asr_brain: rs.adversarial.brain.ASRBrain
      brain object.
   eps: float
      maximum distortion.
   nb_iter: int
      number of iterations.
   eps_iter: float
      attack step size.
   rand_init: (optional bool) 
      random initialization.
   clip_min: (optional) float
      mininum value per input dimension.
   clip_max: (optional) float
      maximum value per input dimension.
   targeted: bool
      if the attack is targeted.
   train_mode_for_backward: bool
      whether to force training mode in backward passes (necessary for RNN models)
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
   Arguments
   ---------
   asr_brain: rs.adversarial.brain.ASRBrain
      brain object.
   eps: float
      maximum distortion.
   nb_iter: int
      number of iterations.
   eps_iter: float
      attack step size.
   rand_init: (optional bool) 
      random initialization.
   clip_min: (optional) float
      mininum value per input dimension.
   clip_max: (optional) float
      maximum value per input dimension.
   targeted: bool
      if the attack is targeted.
   train_mode_for_backward: bool
      whether to force training mode in backward passes (necessary for RNN models)
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
   """
   PGD Attack with order=L2, bounded with Signal-Noise Ratio instead of L2 norm

   Arguments
   ---------
   asr_brain: rs.adversarial.brain.ASRBrain
      brain object.
   snr: float
      maximum distortion.
   nb_iter: int
      number of iterations.
   eps_iter: float
      attack step size.
   rand_init: (optional bool) 
      random initialization.
   clip_min: (optional) float
      mininum value per input dimension.
   clip_max: (optional) float
      maximum value per input dimension.
   targeted: bool
      if the attack is targeted.
   train_mode_for_backward: bool
      whether to force training mode in backward passes (necessary for RNN models)
   """

   def __init__(
         self, asr_brain, snr=40, nb_iter=40,
         rel_eps_iter=0.1, rand_init=True, clip_min=None, clip_max=None,
         targeted=False, train_mode_for_backward=True):
      super(SNRPGDAttack, self).__init__(
         asr_brain=asr_brain, eps=1.0, nb_iter=nb_iter,
         rel_eps_iter=rel_eps_iter, rand_init=rand_init, clip_min=clip_min,
         clip_max=clip_max, targeted=targeted, train_mode_for_backward=train_mode_for_backward)
      assert isinstance(snr, int)
      self.rel_eps = torch.pow(torch.tensor(10.), float(snr)/20)

   def perturb(self, batch):
      save_device = batch.sig[0].device
      batch = batch.to(self.asr_brain.device)
      self.eps = reverse_bound_from_rel_bound(batch, self.rel_eps, ord=2)
      res = super(SNRPGDAttack, self).perturb(batch)
      self.eps = 1.0
      batch.to(save_device)
      return res.to(save_device)

class MaxSNRPGDAttack(ASRLinfPGDAttack):
   """
   PGD Attack with order=Linf, bounded with the Max Signal-Noise Ratio instead of Linf norm

   Arguments
   ---------
   asr_brain: rs.adversarial.brain.ASRBrain
      brain object.
   snr: float
      maximum distortion.
   nb_iter: int
      number of iterations.
   eps_iter: float
      attack step size.
   rand_init: (optional bool) 
      random initialization.
   clip_min: (optional) float
      mininum value per input dimension.
   clip_max: (optional) float
      maximum value per input dimension.
   targeted: bool
      if the attack is targeted.
   train_mode_for_backward: bool
      whether to force training mode in backward passes (necessary for RNN models)
   """

   def __init__(
         self, asr_brain, snr=40, nb_iter=40,
         rel_eps_iter=0.1, rand_init=True, clip_min=None, clip_max=None,
         targeted=False, train_mode_for_backward=True):
      super(MaxSNRPGDAttack, self).__init__(
         asr_brain=asr_brain, eps=1.0, nb_iter=nb_iter,
         rel_eps_iter=rel_eps_iter, rand_init=rand_init, clip_min=clip_min,
         clip_max=clip_max, targeted=targeted, train_mode_for_backward=train_mode_for_backward)
      assert isinstance(snr, int)
      self.rel_eps = torch.pow(torch.tensor(10.), float(snr)/20)

   def perturb(self, batch):
      save_device = batch.sig[0].device
      batch = batch.to(self.asr_brain.device)
      self.eps = reverse_bound_from_rel_bound(batch, self.rel_eps, ord=np.inf)
      res = super(MaxSNRPGDAttack, self).perturb(batch)
      self.eps = 1.0
      batch.to(save_device)
      return res.to(save_device)
