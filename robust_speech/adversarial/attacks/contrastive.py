import torch 
import torch.nn as nn
from advertorch.attacks.utils import rand_init_delta
from advertorch.utils import clamp

import speechbrain as sb
from robust_speech.adversarial.attacks.pgd import ASRPGDAttack, perturb_iterative

class ConstrastiveASRAttack(ASRPGDAttack):
    # TODO : verify that the model supports contrastive loss
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
                
        # fixing the quantized representation of the batch for contrastive adversarial learning 
        _, out, _ = self.asr_brain.compute_forward(batch,stage=sb.stage.VALID)
        q_repr = out.quantized_features, out.codevector_perplexity
        batch.quantized_representation = q_repr

        wav_adv = perturb_iterative(
            batch, self.asr_brain, nb_iter=self.nb_iter,
            eps=self.eps, eps_iter=self.rel_eps_iter*self.eps,
            minimize=self.targeted, ord=self.ord, 
            clip_min=self.clip_min, clip_max=self.clip_max, 
            delta_init=delta, l1_sparsity=self.l1_sparsity
        )
        delattr(batch,'quantized_representation')
        batch.sig = save_input, batch.sig[1]
        batch = batch.to(save_device)
        return wav_adv.data.to(save_device)