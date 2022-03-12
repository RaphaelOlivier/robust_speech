from typing import Tuple, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim

from advertorch.attacks.base import Attack
from advertorch.attacks.base import LabelMixin
from robust_speech.adversarial.attacks.imperceptible import ImperceptibleASRAttack

import speechbrain as sb

import robust_speech as rs


class ASRCarliniWagnerAttack(ImperceptibleASRAttack):
    """
    A Carlini&Wagner attack for ASR models.
    The algorithm follows the first attack in https://arxiv.org/abs/1801.01944
    Based on the ART implementation of Imperceptible (https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/art/attacks/evasion/imperceptible_asr/imperceptible_asr_pytorch.py)

    Arguments
    ---------
     asr_brain : rs.adversarial.brain.ASRBrain
        the brain object to attack
     targeted: bool
        if the attack is targeted (always true for now).
     eps: float
        Linf bound applied to the perturbation.
     learning_rate: float
        the learning rate for the attack algorithm
     max_iter: int
        the maximum number of iterations
     clip_min: float
        mininum value per input dimension (ignored: herefor compatibility).
     clip_max: float
        maximum value per input dimension (ignored: herefor compatibility).
     train_mode_for_backward: bool
        whether to force training mode in backward passes (necessary for RNN models)
    global_max_length: int
        max length of a perturbation
    initial_rescale: float
        initial factor by which to rescale the perturbation
    num_iter_decrease_eps: int
        Number of times to increase epsilon in case of success
    decrease_factor_eps: int
        Factor by which to decrease epsilon in case of failure
    optimizer: Optional["torch.optim.Optimizer"]
        the optimizer to use
    """

    def __init__(
        self,
        asr_brain: rs.adversarial.brain.ASRBrain,
        eps: float = 0.05,
        max_iter: int = 10,
        learning_rate: float = 0.001,
        optimizer: Optional["torch.optim.Optimizer"] = None,
        global_max_length: int = 200000,
        initial_rescale: float = 1.0,
        decrease_factor_eps: float = 0.8,
        num_iter_decrease_eps: int = 1,
        targeted: bool = True,
        train_mode_for_backward: bool = True,
        clip_min: Optional[float] = None,
        clip_max: Optional[float] = None
    ):
        super(ASRCarliniWagnerAttack, self).__init__(
            asr_brain,
            eps=eps,
            max_iter_1=max_iter,
            max_iter_2=0,
            learning_rate_1=learning_rate,
            optimizer_1=optimizer,
            global_max_length=global_max_length,
            initial_rescale=initial_rescale,
            targeted=targeted,
            train_mode_for_backward=train_mode_for_backward,
            clip_min=clip_min,
            clip_max=clip_max
        )
