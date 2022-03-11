


import numpy as np
import torch
import torch.nn as nn
from advertorch.utils import is_float_or_torch_tensor

import speechbrain as sb 

import robust_speech as rs
from robust_speech.adversarial.attacks.pgd import perturb_iterative

from advertorch.attacks.base import Attack,LabelMixin
from robust_speech.adversarial.attacks.attacker import Attacker

class ASRMGAA(Attacker):
    """
    Meta Gradient Adversarial Attack (https://arxiv.org/abs/2108.04204).
    This attack encapsulates another attack (typically PGD) and only retains the part of the perturbation that is relevant for transfable attacks.
    It requires multiple models, i.e. ASR brains.

    Arguments
    ---------
     asr_brain: robust_speech.adversarial.brain.EnsembleAsrBrain
        the brain objects. It should be an EnsembleAsrBrain object where the first brain is the meta model and the second is the train model.
        That second brain is typically also an EnsembleAsrBrain to improve transferability.
     nested_attack_class: robust_speech.adversarial.attacks.attacker.Attacker 
        the nested adversarial attack class.
     nb_iter: int
        number of test (meta) iterations
     eps: float
        bound applied to the meta perturbation.
     ord: int
        order of the attack norm
     clip_min: float
        mininum value per input dimension
     clip_max: float
        maximum value per input dimension
     targeted: bool
        if the attack is targeted
     train_mode_for_backward: bool
        whether to force training mode in backward passes (necessary for RNN models)
        
    """
    
    def __init__(
        self, asr_brain, nested_attack_class, eps=0.3, nb_iter=40,
        rel_eps_iter=1., clip_min=None, clip_max=None,
        ord=np.inf, targeted=False,train_mode_for_backward=True):

        raise NotImplementedError('This attack is under development')

        assert isinstance(asr_brain,rs.adversarial.brain.EnsembleASRBrain) and asr_brain.nmodels==2
        self.nested_attack = nested_attack_class(asr_brain.asr_brains[1])
        self.asr_brain = asr_brain.asr_brains[0]

        self.clip_min = clip_min
        self.clip_max = clip_max
        self.eps = eps
        self.nb_iter = nb_iter
        self.rel_eps_iter = rel_eps_iter
        self.ord = ord
        self.targeted = targeted
        self.train_mode_for_backward = train_mode_for_backward

        assert is_float_or_torch_tensor(self.rel_eps_iter)
        assert is_float_or_torch_tensor(self.eps)

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

        for i in range(self.nb_iter):
            batch.sig = x+delta, batch.sig[1]
            train_adv = self.nested_attack.perturb(batch)
            batch.sig = x, batch.sig[1]
            test_adv = perturb_iterative(
                batch, self.asr_brain, nb_iter=1,
                eps=self.eps, eps_iter=self.rel_eps_iter*self.eps,
                minimize=self.targeted, ord=self.ord, 
                clip_min=self.clip_min, clip_max=self.clip_max, 
                delta_init=nn.Parameter(train_adv - x), l1_sparsity=False
            )
            delta = test_adv - train_adv

        batch.sig = save_input, batch.sig[1]
        batch = batch.to(save_device)
        self.asr_brain.module_eval()
        return (x+delta).data.to(save_device)