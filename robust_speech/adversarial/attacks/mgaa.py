"""
Meta Gradient Adversarial Attack (https://arxiv.org/abs/2108.04204)
"""

import warnings

import numpy as np
import speechbrain as sb
import torch
import torch.nn as nn

import robust_speech as rs
from robust_speech.adversarial.attacks.attacker import Attacker
from robust_speech.adversarial.attacks.pgd import pgd_loop


class ASRMGAA(Attacker):
    """
    Implementation of the Meta Gradient Adversarial Attack
    (https://arxiv.org/abs/2108.04204).
    This attack encapsulates another attack (typically PGD)
    and only retains the part of the perturbation that is relevant for transfable attacks.
    It requires multiple models, i.e. ASR brains.

    Arguments
    ---------
     asr_brain: robust_speech.adversarial.brain.EnsembleAsrBrain
        the brain objects. It should be an EnsembleAsrBrain object
         where the first brain is the meta model
         and the second is the train model.
        That second brain is typically also an EnsembleAsrBrain
         to improve transferability.
     nested_attack_class: robust_speech.adversarial.attacks.attacker.Attacker
        the nested adversarial attack class.
     nb_iter: int
        number of test (meta) iterations
     eps: float
        bound applied to the meta perturbation.
     order: int
        order of the attack norm
     clip_min: float
        mininum value per input dimension
     clip_max: float
        maximum value per input dimension
     targeted: bool
        if the attack is targeted
     train_mode_for_backward: bool
        whether to force training mode in backward passes
        (necessary for RNN models)

    """

    def __init__(
        self,
        asr_brain,
        nested_attack_class,
        eps=0.3,
        nb_iter=40,
        rel_eps_iter=1.0,
        clip_min=None,
        clip_max=None,
        order=np.inf,
        targeted=False,
        train_mode_for_backward=True,
    ):

        warnings.warn(
            "MGAA attack is currently under development. \
            Accurate results are not guaranteed.",
            RuntimeWarning,
        )

        assert (
            isinstance(asr_brain, rs.adversarial.brain.EnsembleASRBrain)
            and asr_brain.nmodels == 2
        )
        self.nested_attack = nested_attack_class(asr_brain.asr_brains[1])
        self.asr_brain = asr_brain.asr_brains[0]

        self.clip_min = clip_min
        self.clip_max = clip_max
        self.eps = eps
        self.nb_iter = nb_iter
        self.rel_eps_iter = rel_eps_iter
        self.order = order
        self.targeted = targeted
        self.train_mode_for_backward = train_mode_for_backward

        assert isinstance(self.rel_eps_iter, torch.Tensor) or isinstance(
            self.rel_eps_iter, float
        )
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
        wav_init = torch.clone(save_input)
        delta = torch.zeros_like(wav_init)

        for _ in range(self.nb_iter):
            batch.sig = wav_init + delta, batch.sig[1]
            train_adv = self.nested_attack.perturb(batch)
            batch.sig = wav_init, batch.sig[1]
            test_adv = pgd_loop(
                batch,
                self.asr_brain,
                nb_iter=1,
                eps=self.eps,
                eps_iter=self.rel_eps_iter * self.eps,
                minimize=self.targeted,
                order=self.order,
                clip_min=self.clip_min,
                clip_max=self.clip_max,
                delta_init=nn.Parameter(train_adv - x),
                l1_sparsity=False,
            )
            delta = test_adv - train_adv

        batch.sig = save_input, batch.sig[1]
        batch = batch.to(save_device)
        self.asr_brain.module_eval()
        return (save_input + delta).data.to(save_device)
