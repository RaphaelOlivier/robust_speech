"""
Carlini&Wagner attack (https://arxiv.org/abs/1801.01944)
"""

from typing import List, Optional, Tuple

import numpy as np
import speechbrain as sb
import torch
import torch.nn as nn
import torch.optim as optim

import robust_speech as rs
from robust_speech.adversarial.attacks.imperceptible import ImperceptibleASRAttack


class ASRCarliniWagnerAttack(ImperceptibleASRAttack):
    """
    A Carlini&Wagner attack for ASR models.
    The algorithm follows the first attack in https://arxiv.org/abs/1801.01944
    Based on the ART implementation of Imperceptible
    (https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/art/attacks/evasion/imperceptible_asr/imperceptible_asr_pytorch.py)

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
        clip_max: Optional[float] = None,
        const: float = 1.0,
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
            clip_max=clip_max,
        )
        self.const = const

    def _forward_1st_stage(
        self,
        original_input: np.ndarray,
        batch: sb.dataio.batch.PaddedBatch,
        local_batch_size: int,
        local_max_length: int,
        rescale: np.ndarray,
        input_mask: np.ndarray,
        real_lengths: np.ndarray,
    ):

        # Compute perturbed inputs
        local_delta = self.global_optimal_delta[:local_batch_size, :local_max_length]
        local_delta_rescale = torch.clamp(local_delta, -self.eps, self.eps).to(
            self.asr_brain.device
        )
        local_delta_rescale *= torch.tensor(rescale).to(self.asr_brain.device)
        adv_input = local_delta_rescale + torch.tensor(original_input).to(
            self.asr_brain.device
        )
        masked_adv_input = adv_input * torch.tensor(input_mask).to(
            self.asr_brain.device
        )

        # Compute loss and decoded output
        batch.sig = masked_adv_input, batch.sig[1]
        predictions = self.asr_brain.compute_forward(batch, rs.Stage.ATTACK)
        loss = self.asr_brain.compute_objectives(predictions, batch, rs.Stage.ATTACK)
        loss = self.const * loss + torch.norm(local_delta_rescale)
        self.asr_brain.module_eval()
        val_predictions = self.asr_brain.compute_forward(batch, sb.Stage.VALID)
        decoded_output = self.asr_brain.get_tokens(val_predictions)
        if self.train_mode_for_backward:
            self.asr_brain.module_train()
        return loss, local_delta, decoded_output, masked_adv_input, local_delta_rescale
