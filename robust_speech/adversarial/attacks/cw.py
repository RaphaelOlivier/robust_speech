
import torch
import torch.nn as nn
import torch.optim as optim

from advertorch.attacks.base import Attack
from advertorch.attacks.base import LabelMixin

import speechbrain as sb

import robust_speech as rs

CARLINI_L2DIST_UPPER = 1e10
CARLINI_COEFF_UPPER = 1e10
INVALID_LABEL = -1
REPEAT_STEP = 10
ONE_MINUS_EPS = 0.999999
UPPER_CHECK = 1e9
PREV_LOSS_INIT = 1e6
NUM_CHECKS = 10

def is_successful(y1, y2, targeted):
    if isinstance(y1,list):
        y1 = torch.tensor(y1)
    if isinstance(y1,torch.Tensor):
        y2=y2.to(y1.device)
        equal = y1.size()==y2.size() and (y1==y2).all()
    else:
        equal = y1 == y2
        if isinstance(equal,torch.Tensor):
            equal=equal.all()
    return targeted == equal

def calc_l2distsq(x, y, mask):
    d = ((x - y)**2) * mask
    return d.view(d.shape[0], -1).sum(dim=1) #/ mask.view(mask.shape[0], -1).sum(dim=1)


class ASRCarliniWagnerAttack(Attack, LabelMixin):
    """
    A Carlini&Wagner attack for ASR models.
    The algorithm follows non strictly the first attack in https://arxiv.org/abs/1801.01944
    This implementation is based on the one in advertorch for classification models (https://github.com/BorealisAI/advertorch/blob/master/advertorch/attacks/carlini_wagner.py).
    It was pruned of elements that aren't present in the ASR CW attack paper above, like tanh rescaling.

    Arguments
    ---------
     asr_brain : rs.adversarial.brain.ASRBrain
        the brain object to attack
     success_only: bool
        if the adversarial noise should only be returned when the attack is successful.
     targeted: bool
        if the attack is targeted (always true for now).
     learning_rate: float
        the learning rate for the attack algorithm
     binary_search_steps: int
        number of binary search times to find the
        optimum
     max_iterations: int
        the maximum number of iterations
     abort_early: bool
        if set to true, abort early if getting stuck in localmin
     initial_const: float
        initial value of the constant c
     eps: float
        Linf bound applied to the perturbation.
     clip_min: float
        mininum value per input dimension (ignored: herefor compatibility).
     clip_max: float
        maximum value per input dimension (ignored: herefor compatibility).
     train_mode_for_backward: bool
        whether to force training mode in backward passes (necessary for RNN models)
    """

    def __init__(self, asr_brain, success_only=True,
                 targeted=True, learning_rate=0.01,
                 binary_search_steps=9, max_iterations=1000,
                 abort_early=True, initial_const=1e0, eps=0.05,
                 clip_min=-1., clip_max=1., train_mode_for_backward=True):
        self.asr_brain = asr_brain
        self.clip_min = clip_min # ignored
        self.clip_max = clip_max # ignored
        self.eps=eps
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.binary_search_steps = binary_search_steps
        self.abort_early = abort_early
        self.initial_const = initial_const
        self.train_mode_for_backward=train_mode_for_backward
        # The last iteration (if we run many steps) repeat the search once.
        self.repeat = binary_search_steps >= REPEAT_STEP
        if not targeted:
            raise NotImplementedError("CW attack is only available for targeted outputs")
        self.targeted = targeted
        self.success_only = success_only

    def _forward_and_update_delta(
            self, optimizer, batch, wavs_init, lens_mask, delta, loss_coeffs):
        optimizer.zero_grad()
        delta = torch.clamp(delta,-self.eps,self.eps)
        adv = delta + wavs_init
        batch.sig = adv,batch.sig[1]
        predictions = self.asr_brain.compute_forward(batch, rs.Stage.ATTACK)
        loss1 = self.asr_brain.compute_objectives(predictions,batch,rs.Stage.ATTACK)
        l2distsq = calc_l2distsq(adv, wavs_init, lens_mask)
        loss = (loss1).sum() + (loss_coeffs * l2distsq).sum()
        loss.backward()
        optimizer.step()

        return loss.item(), l2distsq.data, adv.data

    def _update_if_smaller_dist_succeed(
            self, adv, batch, l2distsq, batch_size,
            cur_l2distsqs, cur_labels, tokens,
            final_l2distsqs, final_labels, final_advs):
        self.asr_brain.module_eval()
        predictions = self.asr_brain.compute_forward(batch, sb.Stage.VALID)
        predicted_tokens = self.asr_brain.get_tokens(predictions)
        if self.train_mode_for_backward:
            self.asr_brain.module_train()
        success = is_successful(predicted_tokens, tokens, self.targeted)
        mask = (l2distsq < cur_l2distsqs) & success
        cur_l2distsqs[mask] = l2distsq[mask]  # redundant
        for i in range(batch_size):
            if mask[i]:
                cur_labels[i] = predicted_tokens[i]

        mask = (l2distsq < final_l2distsqs) & success
        final_l2distsqs[mask] = l2distsq[mask]
        final_advs[mask] = adv[mask]
        for i in range(batch_size):
            if mask[i]:
                final_labels[i] = predicted_tokens[i]

    def _update_unsuccessful(
            self, adv, batch, l2distsq, batch_size,
            final_l2distsqs, final_labels, final_advs):
        mask = (final_l2distsqs==CARLINI_L2DIST_UPPER)
        final_advs[mask] = adv[mask]


    def _update_loss_coeffs(
            self, labs, cur_labels, batch_size, loss_coeffs,
            coeff_upper_bound, coeff_lower_bound):

        for ii in range(batch_size):
            if not is_successful(cur_labels[ii], labs[ii], self.targeted):
                print("Didn't find a successful perturbation: decreasing distance penalty")
                coeff_upper_bound[ii] = min(
                    coeff_upper_bound[ii], loss_coeffs[ii])

                if coeff_upper_bound[ii] < UPPER_CHECK:
                    loss_coeffs[ii] = (coeff_lower_bound[ii] +
                                       coeff_upper_bound[ii]) / 2

            else:
                print("Found a successful perturbation! Increasing distance penalty")
                coeff_lower_bound[ii] = max(
                    coeff_lower_bound[ii], loss_coeffs[ii])
                if coeff_upper_bound[ii] < UPPER_CHECK:
                    loss_coeffs[ii] = (coeff_lower_bound[ii] +
                                       coeff_upper_bound[ii]) / 2

                else:
                    loss_coeffs[ii] *= 10

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
        wavs_init, rel_lengths = batch.sig 
        wavs_init = torch.clone(wavs_init)
        batch_size = wavs_init.size(0)
        if batch_size > 1:
            raise NotImplementedError("%s currently supports only batch size 1"%self.__class__.__name__)
        wav_lengths = (rel_lengths.float()*wavs_init.size(1)).long()
        max_len = wav_lengths.max()
        lens_mask = torch.arange(max_len).to(self.asr_brain.device).expand(len(wav_lengths), max_len) < wav_lengths.unsqueeze(1)
        coeff_lower_bound = wavs_init.new_zeros(batch_size)
        coeff_upper_bound = wavs_init.new_ones(batch_size) * CARLINI_COEFF_UPPER
        loss_coeffs = torch.ones_like(rel_lengths).float() * self.initial_const
        final_l2distsqs = [CARLINI_L2DIST_UPPER] * batch_size
        final_labels = [INVALID_LABEL] * batch_size
        final_advs = torch.clone(wavs_init)
        final_l2distsqs = torch.FloatTensor(final_l2distsqs).to(wavs_init.device)
        # Start binary search
        delta = nn.Parameter(torch.zeros_like(wavs_init))
        for outer_step in range(self.binary_search_steps):
            optimizer = optim.Adam([delta], lr=self.learning_rate)
            cur_l2distsqs = [CARLINI_L2DIST_UPPER] * batch_size
            cur_labels = [INVALID_LABEL] * batch_size
            cur_l2distsqs = torch.FloatTensor(cur_l2distsqs).to(wavs_init.device)
            prevloss = PREV_LOSS_INIT

            if (self.repeat and outer_step == (self.binary_search_steps - 1)):
                loss_coeffs = coeff_upper_bound
            tokens = batch.tokens  
            if isinstance(tokens,sb.dataio.batch.PaddedData):
                tokens = tokens[0]
            for ii in range(self.max_iterations):
                loss, l2distsq, adv = \
                    self._forward_and_update_delta(
                        optimizer, batch, wavs_init, lens_mask, delta, loss_coeffs)
                if self.abort_early:
                    if ii % (self.max_iterations // NUM_CHECKS or 1) == 0:
                        if loss > prevloss * ONE_MINUS_EPS:
                            break
                        prevloss = loss
                adv = torch.clamp(adv - wavs_init,-self.eps,self.eps) + wavs_init
                if ii % 100 == 0:
                    self._update_if_smaller_dist_succeed(
                        adv, batch, l2distsq, batch_size,
                        cur_l2distsqs, cur_labels,tokens,
                        final_l2distsqs, final_labels, final_advs)
            self._update_if_smaller_dist_succeed(
                adv, batch, l2distsq, batch_size,
                cur_l2distsqs, cur_labels,tokens,
                final_l2distsqs, final_labels, final_advs)
            self._update_loss_coeffs(
                tokens, cur_labels, batch_size,
                loss_coeffs, coeff_upper_bound, coeff_lower_bound)
        if not self.success_only:
            self._update_unsuccessful(
                adv, batch, l2distsq, batch_size,
                final_l2distsqs, final_labels, final_advs
            )
        batch.sig = wavs_init, rel_lengths
        batch = batch.to(save_device)
        return final_advs.to(save_device)