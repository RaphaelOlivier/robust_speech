
import torch
import torch.nn as nn
import torch.optim as optim

#from advertorch.utils import calc_l2distsq
from advertorch.utils import torch_arctanh
from advertorch.utils import clamp
from advertorch.utils import to_one_hot
from advertorch.utils import replicate_input
from advertorch.attacks.base import Attack
from advertorch.attacks.base import LabelMixin
from advertorch.attacks.utils import is_successful

import speechbrain as sb

import robust_speech as rs

CARLINI_L2DIST_UPPER = 1e10
CARLINI_COEFF_UPPER = 1e10
INVALID_LABEL = -1
REPEAT_STEP = 10
ONE_MINUS_EPS = 0.999999
UPPER_CHECK = 1e9
PREV_LOSS_INIT = 1e6
TARGET_MULT = 10000.0
NUM_CHECKS = 10

def is_successful(y1, y2, targeted):
    equal = type(y1) == type(y2)
    if equal:
        if isinstance(y1,torch.Tensor):
            equal = y1.size()==y2.size() and (y1==y2).all()
        else:
            equal = y1 == y2
    return targeted == equal

def calc_l2distsq(x, y, mask):
    d = ((x - y)**2) * mask
    return d.view(d.shape[0], -1).sum(dim=1) #/ mask.view(mask.shape[0], -1).sum(dim=1)

def tanh_rescale(x, x_min=-1., x_max=1.):
    return (torch.tanh(x)) * 0.5 * (x_max - x_min) + (x_max + x_min) * 0.5
    #return x


class ASRCarliniWagnerAttack(Attack, LabelMixin):
    """
    The Carlini and Wagner L2 Attack, https://arxiv.org/abs/1608.04644
    :param predict: forward pass function.
    :param targeted: if the attack is targeted.
    :param learning_rate: the learning rate for the attack algorithm
    :param binary_search_steps: number of binary search times to find the
        optimum
    :param max_iterations: the maximum number of iterations
    :param abort_early: if set to true, abort early if getting stuck in local
        min
    :param initial_const: initial value of the constant c
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :param loss_fn: loss function
    """

    def __init__(self, asr_brain, success_only=True,
                 targeted=True, learning_rate=0.01,
                 binary_search_steps=9, max_iterations=1000,
                 abort_early=True, initial_const=1e10,
                 clip_min=-1., clip_max=1.):
        """Carlini Wagner L2 Attack implementation in pytorch."""
        self.asr_brain = asr_brain
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.binary_search_steps = binary_search_steps
        self.abort_early = abort_early
        self.initial_const = initial_const
        # The last iteration (if we run many steps) repeat the search once.
        self.repeat = binary_search_steps >= REPEAT_STEP
        assert targeted, "CW attack only available for targeted outputs"
        self.targeted = targeted
        self.success_only = success_only

    def _forward_and_update_delta(
            self, optimizer, batch, x_atanh, lens_mask, delta, loss_coeffs):
        optimizer.zero_grad()
        adv = tanh_rescale(delta + x_atanh, self.clip_min, self.clip_max)
        transimgs_rescale = tanh_rescale(x_atanh, self.clip_min, self.clip_max)
        batch.sig = adv,batch.sig[1]

        predictions = self.asr_brain.compute_forward(batch, rs.Stage.ATTACK)
        loss1 = self.asr_brain.compute_objectives(predictions,batch,rs.Stage.ATTACK)
        l2distsq = calc_l2distsq(adv, transimgs_rescale, lens_mask)
        loss = (loss_coeffs * loss1).sum() + l2distsq.sum()
        print(loss1)
        loss.backward()
        optimizer.step()

        return loss.item(), l2distsq.data, adv.data

    def _get_arctanh_x(self, x):

        result = clamp((x - self.clip_min) / (self.clip_max - self.clip_min),
                       min=0., max=1.) * 2 - 1
        return torch_arctanh(result * ONE_MINUS_EPS)
        #return x

    def _update_if_smaller_dist_succeed(
            self, adv, batch, l2distsq, batch_size,
            cur_l2distsqs, cur_labels,
            final_l2distsqs, final_labels, final_advs):
        print(adv.norm())
        self.asr_brain.modules.eval()
        predictions = self.asr_brain.compute_forward(batch, sb.Stage.VALID)
        self.asr_brain.modules.train()
        predicted_tokens = predictions[-1]
        predicted_words = [
            self.asr_brain.tokenizer.decode_ids(utt_seq).split(" ")
            for utt_seq in predicted_tokens
        ]
        print(" ".join(predicted_words[0]))
        tokens = batch.tokens
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

        # TODO: remove for loop, not significant, since only called during each
        # binary search step
        for ii in range(batch_size):
            #cur_labels[ii] = int(cur_labels[ii])
            #print(cur_labels[ii], labs[ii])
            if is_successful(cur_labels[ii], labs[ii], self.targeted):
                coeff_upper_bound[ii] = min(
                    coeff_upper_bound[ii], loss_coeffs[ii])

                if coeff_upper_bound[ii] < UPPER_CHECK:
                    loss_coeffs[ii] = (coeff_lower_bound[ii] +
                                       coeff_upper_bound[ii]) / 2

            else:
                coeff_lower_bound[ii] = max(
                    coeff_lower_bound[ii], loss_coeffs[ii])
                if coeff_upper_bound[ii] < UPPER_CHECK:
                    loss_coeffs[ii] = (coeff_lower_bound[ii] +
                                       coeff_upper_bound[ii]) / 2

                else:
                    loss_coeffs[ii] *= 10

    def perturb(self, batch):
        save_device = batch.sig[0].device
        batch = batch.to(self.asr_brain.device)
        wavs_init, rel_lengths = batch.sig 
        wavs_init = replicate_input(wavs_init)
        batch_size = wavs_init.size(0)
        wav_lengths = (rel_lengths.float()*wavs_init.size(1)).long()
        max_len = wav_lengths.max()
        lens_mask = torch.arange(max_len).to(self.asr_brain.device).expand(len(wav_lengths), max_len) < wav_lengths.unsqueeze(1)
        coeff_lower_bound = wavs_init.new_zeros(batch_size)
        coeff_upper_bound = wavs_init.new_ones(batch_size) * CARLINI_COEFF_UPPER
        loss_coeffs = torch.ones_like(rel_lengths).float() * self.initial_const
        final_l2distsqs = [CARLINI_L2DIST_UPPER] * batch_size
        final_labels = [INVALID_LABEL] * batch_size
        final_advs = wavs_init
        x_atanh = self._get_arctanh_x(wavs_init)

        final_l2distsqs = torch.FloatTensor(final_l2distsqs).to(wavs_init.device)
        #final_labels = torch.LongTensor(final_labels).to(wavs_init.device)
        # Start binary search
        for outer_step in range(self.binary_search_steps):
            delta = nn.Parameter(torch.zeros_like(wavs_init))
            optimizer = optim.Adam([delta], lr=self.learning_rate)
            cur_l2distsqs = [CARLINI_L2DIST_UPPER] * batch_size
            cur_labels = [INVALID_LABEL] * batch_size
            cur_l2distsqs = torch.FloatTensor(cur_l2distsqs).to(wavs_init.device)
            #cur_labels = torch.LongTensor(cur_labels).to(wavs_init.device)
            prevloss = PREV_LOSS_INIT

            if (self.repeat and outer_step == (self.binary_search_steps - 1)):
                loss_coeffs = coeff_upper_bound
            for ii in range(self.max_iterations):
                loss, l2distsq, adv = \
                    self._forward_and_update_delta(
                        optimizer, batch, x_atanh, lens_mask, delta, loss_coeffs)
                if self.abort_early:
                    if ii % (self.max_iterations // NUM_CHECKS or 1) == 0:
                        if loss > prevloss * ONE_MINUS_EPS:
                            break
                        prevloss = loss
                self._update_if_smaller_dist_succeed(
                    adv, batch, l2distsq, batch_size,
                    cur_l2distsqs, cur_labels,
                    final_l2distsqs, final_labels, final_advs)
            self._update_loss_coeffs(
                batch.tokens, cur_labels, batch_size,
                loss_coeffs, coeff_upper_bound, coeff_lower_bound)
        if not self.success_only:
            self._update_unsuccessful(
                adv, batch, l2distsq, batch_size,
                final_l2distsqs, final_labels, final_advs
            )
        batch.sig = wavs_init, rel_lengths
        batch = batch.to(save_device)
        return final_advs.to(save_device)