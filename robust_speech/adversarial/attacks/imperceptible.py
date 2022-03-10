"""
An implementation of the Imperceptible ASR attack (https://arxiv.org/abs/1903.10346).
Based on a mixture of the Advertorch CW attack (https://github.com/BorealisAI/advertorch/blob/master/advertorch/attacks/carlini_wagner.py)
and the ART implementation of Imperceptible (https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/art/attacks/evasion/imperceptible_asr/imperceptible_asr_pytorch.py)
This attack is currently not achieving its expected results and is under debugging.
"""


from typing import Tuple, List
import math
import numpy as np
from scipy.signal import argrelextrema
import torch
import torch.nn as nn
import torch.optim as optim

import speechbrain as sb

import robust_speech as rs
from robust_speech.adversarial.attacks.cw import (
    ASRCarliniWagnerAttack,
    CARLINI_L2DIST_UPPER,
    CARLINI_COEFF_UPPER,
    INVALID_LABEL,
    REPEAT_STEP,
    ONE_MINUS_EPS,
    PREV_LOSS_INIT,
    NUM_CHECKS 
)

class ImperceptibleASRAttack(ASRCarliniWagnerAttack):
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
     abort_early: bool
        if set to true, abort early if getting stuck in localmin
     eps: float
        Linf bound applied to the perturbation.
     learning_rate_phase_1: float
        the learning rate for the attack algorithm in phase 1
     max_iterations_phase_1: int
        the maximum number of iterations in phase 1
     learning_rate_phase_2: float
        the learning rate for the attack algorithm in phase 2
     max_iterations_phase_2: int
        the maximum number of iterations in phase 2
     binary_search_steps_phase_2: int
        number of binary search times to find the optimum in phase 2
     initial_const_phase_2: float
        initial value of the constant c in phase 2
     clip_min: float
        mininum value per input dimension (ignored: herefor compatibility).
     clip_max: float
        maximum value per input dimension (ignored: herefor compatibility).
     train_mode_for_backward: bool
        whether to force training mode in backward passes (necessary for RNN models)
     win_length: int
        window length for computing spectral density
     hop_length: int
        hop length for computing spectral density
     n_fft: int
        number of FFT bins for computing spectral density.
    """
    def __init__(
        self, asr_brain, success_only=True,abort_early=True, targeted=True,eps=0.05,
        learning_rate_phase_1=0.01, max_iterations_phase_1=1000,
        learning_rate_phase_2=0.01, max_iterations_phase_2=1000,
        binary_search_steps_phase_2=9, initial_const_phase_2=1e0,
        clip_min=-1., clip_max=1., train_mode_for_backward=True,
        win_length = 2048,hop_length = 512,n_fft = 2048,
        ):


        self.asr_brain = asr_brain
        self.clip_min = clip_min # ignored
        self.clip_max = clip_max # ignored
        self.abort_early = abort_early
        assert targeted, "CW attack only available for targeted outputs"
        self.targeted = targeted
        self.success_only = success_only
        self.train_mode_for_backward=train_mode_for_backward
        self.eps=eps
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_fft = n_fft

        self.learning_rate = learning_rate_phase_1
        self.max_iterations = max_iterations_phase_1
        self.binary_search_steps = 1
        self.initial_const = 0.
        self.repeat = False

        self.learning_rate_2 = learning_rate_phase_2
        self.max_iterations_2 = max_iterations_phase_2
        self.binary_search_steps_2 = binary_search_steps_phase_2
        self.initial_const_2 = initial_const_phase_2
        self.repeat_2 = binary_search_steps_phase_2 >= REPEAT_STEP
        
        raise NotImplementedError('This attack is under development')

    def _forward_and_update_delta_phase_2(
            self, optimizer, batch, wavs_init, wav_lengths, delta, loss_coeffs, theta, max_psd_init):
        optimizer.zero_grad()
        delta = torch.clamp(delta,-self.eps,self.eps)
        adv = delta + wavs_init
        batch.sig = adv,batch.sig[1]
        predictions = self.asr_brain.compute_forward(batch, rs.Stage.ATTACK)
        loss1 = self.asr_brain.compute_objectives(predictions,batch,rs.Stage.ATTACK)
        l2distsq = self._calc_psd_penalty(delta, theta, max_psd_init, wav_lengths)
        
        loss = (loss1).sum() + (loss_coeffs * l2distsq).sum()
        loss.backward()
        optimizer.step()

        return loss.item(), l2distsq.data, adv.data

    def _calc_psd_penalty(
        self,
        delta: "torch.Tensor",
        theta: List[np.ndarray],
        max_psd_init: List[np.ndarray],
        wav_lengths,
    ) -> "torch.Tensor":
        """The forward pass of the second stage of the attack.
        """

        # Compute loss for masking threshold
        losses = []
        relu = torch.nn.ReLU()

        for i, _ in enumerate(theta):
            psd_transform_delta = self._psd_transform(
                delta=delta[i, : wav_lengths[i]], max_psd_init=max_psd_init[i]
            )
            loss = torch.mean(relu(psd_transform_delta -theta[i].to(self.asr_brain.device)))
            losses.append(loss)

        losses_stack = torch.stack(losses)

        return losses_stack

    def _compute_masking_threshold(self, x: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
        """Compute the masking threshold and the maximum psd of the original audio.
        """

        # First compute the psd matrix
        # Get window for the transformation
        window = torch.hann_window(self.win_length, periodic=True).to(self.asr_brain.device)

        # Do transformation
        transformed_x = torch.stft(
            input=x, n_fft=self.n_fft, hop_length=self.hop_length,
             win_length=self.win_length, window=window, center=False, return_complex=True
        )
        transformed_x *= math.sqrt(8.0 / 3.0)
        psd = torch.abs(transformed_x / self.win_length)
        original_max_psd = torch.max(psd * psd)
        with np.errstate(divide="ignore"):
            psd = (20 * torch.log10(psd)).clip(min=-200)
        psd = 96 - torch.max(psd) + psd

        # Compute freqs and barks
        freqs = torch.fft.rfftfreq(n=self.n_fft, d = 1./16000)
        barks = 13 * torch.arctan(0.00076 * freqs) + 3.5 * torch.arctan(torch.pow(freqs / 7500.0, 2))
        
        # Compute quiet threshold
        ath = torch.zeros(len(barks), dtype=torch.float32) - np.inf
        bark_idx = torch.argmax((barks > 1).long())
        ath[bark_idx:] = (
            3.64 * torch.pow(freqs[bark_idx:] * 0.001, -0.8)
            - 6.5 * np.exp(-0.6 * torch.pow(0.001 * freqs[bark_idx:] - 3.3, 2))
            + 0.001 * torch.pow(0.001 * freqs[bark_idx:], 4)
            - 12
        )

        # Compute the global masking threshold theta
        theta = []
        for i in range(psd.size(1)):
            # Compute masker index
            masker_idx = argrelextrema(psd[:, i].detach().cpu().numpy(), np.greater)[0]
            if 0 in masker_idx:
                masker_idx = np.delete(masker_idx, 0)
            if len(psd[:, i]) - 1 in masker_idx:
                masker_idx = np.delete(masker_idx, len(psd[:, i].numpy() - 1))
            barks_psd = torch.zeros([len(masker_idx), 3], dtype=torch.float32)
            barks_psd[:, 0] = barks[masker_idx]
            barks_psd[:, 1] = 10 * torch.log10(
                torch.pow(10, psd[:, i][masker_idx - 1] / 10.0)
                + torch.pow(10, psd[:, i][masker_idx] / 10.0)
                + torch.pow(10, psd[:, i][masker_idx + 1] / 10.0)
            )
            barks_psd[:, 2] = torch.tensor(masker_idx)

            for j in range(len(masker_idx)):
                if barks_psd.shape[0] <= j + 1:
                    break

                while barks_psd[j + 1, 0] - barks_psd[j, 0] < 0.5:
                    quiet_threshold = (
                        3.64 * torch.pow(freqs[int(barks_psd[j, 2])] * 0.001, -0.8)
                        - 6.5 * torch.exp(-0.6 * torch.pow(0.001 * freqs[int(barks_psd[j, 2])] - 3.3, 2))
                        + 0.001 * torch.pow(0.001 * freqs[int(barks_psd[j, 2])], 4)
                        - 12
                    )
                    if barks_psd[j, 1] < quiet_threshold:
                        barks_psd = np.delete(barks_psd, j, axis=0)

                    if barks_psd.shape[0] == j + 1:
                        break

                    if barks_psd[j, 1] < barks_psd[j + 1, 1]:
                        barks_psd = np.delete(barks_psd, j, axis=0)
                    else:
                        barks_psd = np.delete(barks_psd, j + 1, axis=0)

                    if barks_psd.shape[0] == j + 1:
                        break

            # Compute the global masking threshold
            delta = 1 * (-6.025 - 0.275 * barks_psd[:, 0])

            t_s = []
            for m in range(barks_psd.shape[0]):
                d_z = barks - barks_psd[m, 0]
                zero_idx = np.argmax(d_z > 0)
                s_f = np.zeros(len(d_z), dtype=np.float32)
                s_f[:zero_idx] = 27 * d_z[:zero_idx]
                s_f[zero_idx:] = (-27 + 0.37 * max(barks_psd[m, 1] - 40, 0)) * d_z[zero_idx:]
                t_s.append(barks_psd[m, 1] + delta[m] + s_f)

            t_s_array = torch.stack(t_s,dim=0)
            theta.append(torch.sum(torch.pow(10, t_s_array / 10.0), dim=0) + torch.pow(10, ath / 10.0))

        theta = torch.stack(theta,dim=0)
        return theta, original_max_psd

    def _psd_transform(self, delta: "torch.Tensor", max_psd_init: "torch.Tensor") -> "torch.Tensor":
        """Compute the psd matrix of the perturbation.
        """

        # Get window for the transformation
        window_fn = torch.hann_window  # type: ignore

        # Return STFT of delta
        delta_stft = torch.stft(
            delta,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            center=False,
            window=window_fn(self.win_length).to(self.asr_brain.device),
            return_complex=False
        ).to(self.asr_brain.device)

        # Take abs of complex STFT results
        transformed_delta = torch.sqrt(torch.sum(torch.square(delta_stft), -1))
        # Compute the psd matrix
        psd = (8.0 / 3.0) * transformed_delta / self.win_length
        psd = psd ** 2
        psd = (
            torch.pow(torch.tensor(10.0).type(torch.float32), torch.tensor(9.6).type(torch.float32)).to(
                self.asr_brain.device
            )
            / torch.reshape(max_psd_init.to(self.asr_brain.device), [-1, 1, 1])
            * psd.type(torch.float32)
        )

        return psd

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
        
        theta_batch = []
        max_psd_init = []

        save_device = batch.sig[0].device
        batch = batch.to(self.asr_brain.device)
        wavs_init, rel_lengths = batch.sig 
        wavs_init = torch.clone(wavs_init)
        batch_size = wavs_init.size(0)

        for _, x_i in enumerate(wavs_init):
            theta, original_max_psd = self._compute_masking_threshold(x_i)
            theta = theta.transpose(1, 0)
            theta_batch.append(theta)
            max_psd_init.append(original_max_psd)


        advs_phase_1 = ASRCarliniWagnerAttack.perturb(self,batch)

        if batch_size > 1:
            raise NotImplementedError("CW attack currently supports only batch size 1")
        wav_lengths = (rel_lengths.float()*wavs_init.size(1)).long()
        max_len = wav_lengths.max()
        coeff_lower_bound = wavs_init.new_zeros(batch_size)
        coeff_upper_bound = wavs_init.new_ones(batch_size) * CARLINI_COEFF_UPPER
        loss_coeffs = torch.ones_like(rel_lengths).float() * self.initial_const_2
        final_l2distsqs = [CARLINI_L2DIST_UPPER] * batch_size
        final_labels = [INVALID_LABEL] * batch_size
        final_advs = torch.clone(wavs_init)

        final_l2distsqs = torch.FloatTensor(final_l2distsqs).to(wavs_init.device)

        
        # Start binary search
        delta = nn.Parameter(torch.clone(advs_phase_1-wavs_init))
        print(torch.norm(delta))
        for outer_step in range(self.binary_search_steps_2):
            optimizer = optim.Adam([delta], lr=self.learning_rate_2)
            cur_l2distsqs = [CARLINI_L2DIST_UPPER] * batch_size
            cur_labels = [INVALID_LABEL] * batch_size
            cur_l2distsqs = torch.FloatTensor(cur_l2distsqs).to(wavs_init.device)
            prevloss = PREV_LOSS_INIT

            if (self.repeat_2 and outer_step == (self.binary_search_steps_2 - 1)):
                loss_coeffs = coeff_upper_bound
            tokens = batch.tokens  
            if isinstance(tokens,sb.dataio.batch.PaddedData):
                tokens = tokens[0]
            for ii in range(self.max_iterations_2):
                loss, l2distsq, adv = \
                    self._forward_and_update_delta_phase_2(
                        optimizer, batch, wavs_init, wav_lengths, delta, loss_coeffs, theta_batch, max_psd_init)
                if self.abort_early:
                    if ii % (self.max_iterations_2 // NUM_CHECKS or 1) == 0:
                        if loss > prevloss * ONE_MINUS_EPS:
                            break
                        prevloss = loss
                adv = torch.clamp(adv - wavs_init,-self.eps,self.eps) + wavs_init
                if ii % 100 == 0:
                    self._update_if_smaller_dist_succeed(
                        adv, batch, l2distsq, batch_size,
                        cur_l2distsqs, cur_labels,tokens,
                        final_l2distsqs, final_labels, final_advs)
                    print(loss, l2distsq)
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