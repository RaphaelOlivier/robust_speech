"""
Wav2Vec2 specific attacks
"""

import torch
import torch.nn as nn
from robust_speech.adversarial.utils import rand_assign
import speechbrain as sb
from robust_speech.adversarial.attacks.pgd import ASRPGDAttack, pgd_loop
from robust_speech.models.wav2vec2_pretrain import AdvHuggingFaceWav2Vec2Pretrain
import robust_speech as rs


class ContrastiveASRAttack(ASRPGDAttack):
    """
    Implementation of a Contrastive attack for Wav2Vec2.
    This attack is inspired by Adversarial Contrastive Learning for self-supervised Classification
    (https://arxiv.org/abs/2006.07589)
    It modifies inputs in order to mismatch context c(x+delta) from quantized representation q(x)

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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self._check_for_contrastive_loss():
            raise ValueError(
                "Contrastive attack can only be applied to wav2vec2-based models that support fixing quantized representations"
            )

    def _check_for_contrastive_loss(self):
        if not hasattr(self.asr_brain.modules, "wav2vec2"):
            return False
        if not isinstance(self.asr_brain.modules.wav2vec2, AdvHuggingFaceWav2Vec2Pretrain):
            return False
        return True

    def perturb(self, batch):
        if self.train_mode_for_backward:
            self.asr_brain.module_train()
        else:
            self.asr_brain.module_eval()
        """
        Given an audio batch, returns its adversarial counterpart with
        an attack radius of eps.
        
        Arguments
        ---------
        batch: PaddedBatch

        Returns
        -------
        tensor containing perturbed inputs.
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
            rand_assign(
                delta, self.ord, self.eps)
            delta.data = torch.clamp(
                x + delta.data, min=self.clip_min, max=self.clip_max) - x

        # fixing the quantized representation of the batch for contrastive adversarial learning
        _, out, _ = self.asr_brain.compute_forward(batch, stage=sb.Stage.VALID)
        q_repr = out.projected_quantized_states.detach(), out.codevector_perplexity.detach()
        batch.quantized_representation = q_repr
        wav_adv = pgd_loop(
            batch, self.asr_brain, nb_iter=self.nb_iter,
            eps=self.eps, eps_iter=self.rel_eps_iter*self.eps,
            minimize=self.targeted, ord=self.ord,
            clip_min=self.clip_min, clip_max=self.clip_max,
            delta_init=delta, l1_sparsity=self.l1_sparsity
        )
        # delattr(batch,'quantized_representation')
        batch.sig = save_input, batch.sig[1]
        batch = batch.to(save_device)
        self.asr_brain.module_eval()
        return wav_adv.data.to(save_device)


class ASRFeatureAdversary(ASRPGDAttack):
    """
    Implementation of an attack for Wav2Vec2.
    This attack tries to maximize the L2 distance between the context of the natural and adversarial input.
    This makes it somehow similar to a Feature adversary (https://arxiv.org/pdf/1511.05122.pdf) but in an untargeted way

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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self._check_for_contrastive_loss():
            raise ValueError(
                "Feature Adversary attack can only be applied to wav2vec2-based models that support fixing quantized representations"
            )

    def _check_for_contrastive_loss(self):
        if not hasattr(self.asr_brain.modules, "wav2vec2"):
            return False
        if not isinstance(self.asr_brain.modules.wav2vec2, AdvHuggingFaceWav2Vec2Pretrain):
            return False
        return True

    def perturb(self, batch):
        if self.train_mode_for_backward:
            self.asr_brain.module_train()
        else:
            self.asr_brain.module_eval()
        """
        Given an audio batch, returns its adversarial counterpart with
        an attack radius of eps.
        
        Arguments
        ---------
        batch: PaddedBatch

        Returns
        -------
        tensor containing perturbed inputs.
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
            rand_assign(
                delta, self.ord, self.eps)
            delta.data = torch.clamp(
                x + delta.data, min=self.clip_min, max=self.clip_max) - x

        # fixing the quantized representation of the batch for contrastive adversarial learning
        _, out, _ = self.asr_brain.compute_forward(batch, stage=sb.Stage.VALID)
        q_repr = out.projected_quantized_states.detach(), out.codevector_perplexity.detach()
        batch.quantized_representation = q_repr

        class NestedClassForFeatureAdversary:
            def __init__(self, wav2vec2, batch):
                self.wav2vec2 = wav2vec2
                self.init_features = self.wav2vec2(batch.sig[0])[0].detach()

            def compute_forward(self, batch, stage):
                assert stage == rs.Stage.ATTACK
                features = self.wav2vec2(batch.sig[0])[0]
                return features, None

            def compute_objectives(self, predictions, batch, stage):
                assert stage == rs.Stage.ATTACK
                loss = torch.square(predictions[0]-self.init_features).sum()
                return loss

        wav_adv = pgd_loop(
            batch, NestedClassForFeatureAdversary(self.asr_brain.modules.wav2vec2.model.wav2vec2, batch), nb_iter=self.nb_iter,
            eps=self.eps, eps_iter=self.rel_eps_iter*self.eps,
            minimize=self.targeted, ord=self.ord,
            clip_min=self.clip_min, clip_max=self.clip_max,
            delta_init=delta, l1_sparsity=self.l1_sparsity
        )
        # delattr(batch,'quantized_representation')
        batch.sig = save_input, batch.sig[1]
        batch = batch.to(save_device)
        self.asr_brain.module_eval()
        return wav_adv.data.to(save_device)
