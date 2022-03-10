import torch 
import torch.nn as nn
from advertorch.attacks.utils import rand_init_delta
from advertorch.utils import clamp
import speechbrain as sb
from robust_speech.adversarial.attacks.pgd import ASRPGDAttack, perturb_iterative
from robust_speech.models.wav2vec2_pretrain import AdvHuggingFaceWav2Vec2Pretrain

class ConstrastiveASRAttack(ASRPGDAttack):
    """
    Implementation of the Contrastive attack for Wav2Vec2.
    This attack is inspired by Adversarial Contrastive Learning for self-supervised Classification
    (https://arxiv.org/abs/2006.07589)
    
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
        if not hasattr(self.asr_brain.modules,"wav2vec2"):
            return False 
        if not isinstance(self.asr_brain.modules.wav2vec2,AdvHuggingFaceWav2Vec2Pretrain):
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
            rand_init_delta(
                delta, x, self.ord, self.eps, clip_min, clip_max)
            delta.data = clamp(
                x + delta.data, min=self.clip_min, max=self.clip_max) - x
                
        # fixing the quantized representation of the batch for contrastive adversarial learning 
        _, out, _ = self.asr_brain.compute_forward(batch,stage=sb.Stage.VALID)
        q_repr = out.projected_quantized_states.detach(), out.codevector_perplexity.detach()
        batch.quantized_representation = q_repr
        h = batch.sig[0][:,None]
        wav_adv = perturb_iterative(
            batch, self.asr_brain, nb_iter=self.nb_iter,
            eps=self.eps, eps_iter=self.rel_eps_iter*self.eps,
            minimize=self.targeted, ord=self.ord, 
            clip_min=self.clip_min, clip_max=self.clip_max, 
            delta_init=delta, l1_sparsity=self.l1_sparsity
        )
        #delattr(batch,'quantized_representation')
        batch.sig = save_input, batch.sig[1]
        batch = batch.to(save_device)
        return wav_adv.data.to(save_device)