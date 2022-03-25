"""
Imperceptible ASR attack (https://arxiv.org/abs/1903.10346)
"""

import math
from typing import List, Optional, Tuple

import numpy as np
import speechbrain as sb
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.signal import argrelextrema
from torch.autograd import Variable

import robust_speech as rs
from robust_speech.adversarial.attacks.attacker import Attacker


class ImperceptibleASRAttack(Attacker):
    """
    An implementation of the Imperceptible ASR attack
    (https://arxiv.org/abs/1903.10346).
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
     learning_rate_1: float
        the learning rate for the attack algorithm in phase 1
     max_iter_1: int
        the maximum number of iterations in phase 1
     learning_rate_2: float
        the learning rate for the attack algorithm in phase 2
     max_iter_2: int
        the maximum number of iterations in phase 2
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
    global_max_length: int
        max length of a perturbation
    initial_rescale: float
        initial factor by which to rescale the perturbation
    num_iter_decrease_eps: int
        Number of times to increase epsilon in case of success
    decrease_factor_eps: int
        Factor by which to decrease epsilon in case of failure
    alpha: float
        regularization constant for threshold loss term
    num_iter_decrease_alpha: int
        Number of times to decrease alpha in case of failure
    num_iter_decrease_alpha: int
        Number of times to increase alpha in case of success
    decrease_factor_alpha: int
        Factor by which to decrease alpha in case of failure
    increase_factor_alpha: int
        Factor by which to increase alpha in case of success
    optimizer_1: Optional["torch.optim.Optimizer"]
        the optimizer to use in phase 1
    optimizer_2: Optional["torch.optim.Optimizer"]
        the optimizer to use in phase 2
    """

    def __init__(
        self,
        asr_brain: rs.adversarial.brain.ASRBrain,
        eps: float = 0.05,
        max_iter_1: int = 10,
        max_iter_2: int = 4000,
        learning_rate_1: float = 0.001,
        learning_rate_2: float = 5e-4,
        optimizer_1: Optional["torch.optim.Optimizer"] = None,
        optimizer_2: Optional["torch.optim.Optimizer"] = None,
        global_max_length: int = 200000,
        initial_rescale: float = 1.0,
        decrease_factor_eps: float = 0.8,
        num_iter_decrease_eps: int = 1,
        alpha: float = 1.2,
        increase_factor_alpha: float = 1.2,
        num_iter_increase_alpha: int = 20,
        decrease_factor_alpha: float = 0.8,
        num_iter_decrease_alpha: int = 20,
        win_length: int = 2048,
        hop_length: int = 512,
        n_fft: int = 2048,
        targeted: bool = True,
        train_mode_for_backward: bool = True,
        clip_min: Optional[float] = None,
        clip_max: Optional[float] = None,
    ):

        self.asr_brain = asr_brain
        self.eps = eps
        self.max_iter_1 = max_iter_1
        self.max_iter_2 = max_iter_2
        self.learning_rate_1 = learning_rate_1
        self.learning_rate_2 = learning_rate_2
        self.global_max_length = global_max_length
        self.initial_rescale = initial_rescale
        self.decrease_factor_eps = decrease_factor_eps
        self.num_iter_decrease_eps = num_iter_decrease_eps
        self.alpha = alpha
        self.increase_factor_alpha = increase_factor_alpha
        self.num_iter_increase_alpha = num_iter_increase_alpha
        self.decrease_factor_alpha = decrease_factor_alpha
        self.num_iter_decrease_alpha = num_iter_decrease_alpha
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_fft = n_fft

        self.clip_min = clip_min  # ignored
        self.clip_max = clip_max  # ignored
        assert targeted, (
            "%s attack only available for targeted outputs" % self.__class__.__name__
        )
        self.targeted = targeted
        self.train_mode_for_backward = train_mode_for_backward
        self._optimizer_arg_1 = optimizer_1
        self._optimizer_arg_2 = optimizer_2

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
        batch.sig = wav_init, batch.sig[1]
        # First reset delta
        global_optimal_delta = torch.zeros(batch.batchsize, self.global_max_length).to(
            self.asr_brain.device
        )
        self.global_optimal_delta = nn.Parameter(global_optimal_delta)
        # Next, reset optimizers
        if self._optimizer_arg_1 is None:
            self.optimizer_1 = torch.optim.Adam(
                params=[self.global_optimal_delta], lr=self.learning_rate_1
            )
        else:
            self.optimizer_1 = self._optimizer_arg_1(  # type: ignore
                params=[self.global_optimal_delta], lr=self.learning_rate_1
            )

        if self._optimizer_arg_2 is None:
            self.optimizer_2 = torch.optim.Adam(
                params=[self.global_optimal_delta], lr=self.learning_rate_2
            )
        else:
            self.optimizer_2 = self._optimizer_arg_2(  # type: ignore
                params=[self.global_optimal_delta], lr=self.learning_rate_2
            )

        # Then compute the batch
        adv_wav = self._generate_batch(batch)

        batch.sig = save_input, batch.sig[1]
        batch = batch.to(save_device)
        self.asr_brain.module_eval()
        return adv_wav

    def _generate_batch(self, batch):
        """
        Run all attack stages

        Arguments
        ---------
        batch : sb.PaddedBatch
           The input batch to perturb

        Returns
        -------
        the tensor of the perturbed batch
        """
        # First stage of attack
        original_input = torch.clone(batch.sig[0])
        successful_adv_input_1st_stage = self._attack_1st_stage(batch)
        successful_perturbation_1st_stage = (
            successful_adv_input_1st_stage - original_input
        )

        if self.max_iter_2 == 0:
            return successful_adv_input_1st_stage

        # Compute original masking threshold and maximum psd
        theta_batch = []
        original_max_psd_batch = []
        wav_init = batch.sig[0]
        lengths = (wav_init.size(1) * batch.sig[1]).long()
        wavs = [wav_init[i, : lengths[i]] for i in range(batch.batchsize)]
        for _, wav_i in enumerate(wavs):
            theta, original_max_psd = None, None
            theta, original_max_psd = self._compute_masking_threshold(wav_i)
            theta = theta.transpose(1, 0)
            theta_batch.append(theta)
            original_max_psd_batch.append(original_max_psd)

        # Reset delta with new result
        local_batch_shape = successful_adv_input_1st_stage.shape
        self.global_optimal_delta.data = torch.zeros(
            batch.batchsize, self.global_max_length
        ).to(self.asr_brain.device)
        self.global_optimal_delta.data[
            : local_batch_shape[0], : local_batch_shape[1]
        ] = successful_perturbation_1st_stage

        # Second stage of attack
        successful_adv_input_2nd_stage = self._attack_2nd_stage(
            batch,
            theta_batch=theta_batch,
            original_max_psd_batch=original_max_psd_batch,
        )

        return successful_adv_input_2nd_stage

    def _attack_1st_stage(self, batch) -> Tuple["torch.Tensor", np.ndarray]:
        """
        The first stage of the attack.
        """
        # Compute local shape
        local_batch_size = batch.batchsize
        real_lengths = (
            (batch.sig[1] * batch.sig[0].size(1)).long().detach().cpu().numpy()
        )
        local_max_length = np.max(real_lengths)

        # Initialize rescale
        rescale = (
            np.ones([local_batch_size, local_max_length], dtype=np.float32)
            * self.initial_rescale
        )

        # Reformat input
        input_mask = np.zeros([local_batch_size, local_max_length], dtype=np.float32)
        original_input = torch.clone(batch.sig[0])
        for local_batch_size_idx in range(local_batch_size):
            input_mask[local_batch_size_idx, : real_lengths[local_batch_size_idx]] = 1

        # Optimization loop
        successful_adv_input: List[Optional["torch.Tensor"]] = [None] * local_batch_size
        trans = [None] * local_batch_size

        for iter_1st_stage_idx in range(self.max_iter_1):
            # Zero the parameter gradients
            self.optimizer_1.zero_grad()

            # Call to forward pass
            (
                loss,
                local_delta,
                decoded_output,
                masked_adv_input,
                _,
            ) = self._forward_1st_stage(
                original_input=original_input,
                batch=batch,
                local_batch_size=local_batch_size,
                local_max_length=local_max_length,
                rescale=rescale,
                input_mask=input_mask,
                real_lengths=real_lengths,
            )
            loss.backward()

            # Get sign of the gradients
            self.global_optimal_delta.grad = torch.sign(self.global_optimal_delta.grad)

            # Do optimization
            self.optimizer_1.step()

            # Save the best adversarial example and adjust the rescale
            # coefficient if successful
            if iter_1st_stage_idx % self.num_iter_decrease_eps == 0:
                for local_batch_size_idx in range(local_batch_size):
                    tokens = (
                        batch.tokens[local_batch_size_idx]
                        .detach()
                        .cpu()
                        .numpy()
                        .reshape(-1)
                    )
                    pred = np.array(decoded_output[local_batch_size_idx]).reshape(-1)
                    if len(pred) == len(tokens) and (pred == tokens).all():
                        print("Found one")
                        # Adjust the rescale coefficient
                        max_local_delta = np.max(
                            np.abs(
                                local_delta[local_batch_size_idx].detach().cpu().numpy()
                            )
                        )

                        if (
                            rescale[local_batch_size_idx][0] * self.eps
                            > max_local_delta
                        ):
                            rescale[local_batch_size_idx] = max_local_delta / self.eps
                        rescale[local_batch_size_idx] *= self.decrease_factor_eps

                        # Save the best adversarial example
                        successful_adv_input[local_batch_size_idx] = masked_adv_input[
                            local_batch_size_idx
                        ]
                        trans[local_batch_size_idx] = decoded_output[
                            local_batch_size_idx
                        ]

            # If attack is unsuccessful
            if iter_1st_stage_idx == self.max_iter_1 - 1:
                for local_batch_size_idx in range(local_batch_size):
                    if successful_adv_input[local_batch_size_idx] is None:
                        successful_adv_input[local_batch_size_idx] = masked_adv_input[
                            local_batch_size_idx
                        ]
                        trans[local_batch_size_idx] = decoded_output[
                            local_batch_size_idx
                        ]

        result = torch.stack(successful_adv_input)  # type: ignore

        batch.sig = original_input, batch.sig[1]
        return result

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
        self.asr_brain.module_eval()
        val_predictions = self.asr_brain.compute_forward(batch, sb.Stage.VALID)
        decoded_output = self.asr_brain.get_tokens(val_predictions)
        if self.train_mode_for_backward:
            self.asr_brain.module_train()
        return loss, local_delta, decoded_output, masked_adv_input, local_delta_rescale

    def _attack_2nd_stage(
        self,
        batch,
        theta_batch: List[np.ndarray],
        original_max_psd_batch: List[np.ndarray],
    ):

        # Compute local shape
        local_batch_size = batch.batchsize
        real_lengths = (
            (batch.sig[1] * batch.sig[0].size(1)).long().detach().cpu().numpy()
        )
        local_max_length = np.max(real_lengths)

        # Initialize rescale
        rescale = (
            np.ones([local_batch_size, local_max_length], dtype=np.float32)
            * self.initial_rescale
        )

        # Reformat input
        input_mask = np.zeros([local_batch_size, local_max_length], dtype=np.float32)
        original_input = torch.clone(batch.sig[0])
        for local_batch_size_idx in range(local_batch_size):
            input_mask[local_batch_size_idx, : real_lengths[local_batch_size_idx]] = 1

        # Optimization loop
        successful_adv_input: List[Optional["torch.Tensor"]] = [None] * local_batch_size
        trans = [None] * local_batch_size

        # Initialize alpha and rescale
        alpha = np.array([self.alpha] * local_batch_size, dtype=np.float32)
        rescale = (
            np.ones([local_batch_size, local_max_length], dtype=np.float32)
            * self.initial_rescale
        )

        # Reformat input
        input_mask = np.zeros([local_batch_size, local_max_length], dtype=np.float32)
        original_input = torch.clone(batch.sig[0])
        for local_batch_size_idx in range(local_batch_size):
            input_mask[local_batch_size_idx, : real_lengths[local_batch_size_idx]] = 1

        # Optimization loop
        successful_adv_input: List[Optional["torch.Tensor"]] = [None] * local_batch_size
        best_loss_2nd_stage = [np.inf] * local_batch_size
        trans = [None] * local_batch_size

        for iter_2nd_stage_idx in range(self.max_iter_2):
            # Zero the parameter gradients
            self.optimizer_2.zero_grad()

            # Call to forward pass of the first stage
            (
                loss_1st_stage,
                _,
                decoded_output,
                masked_adv_input,
                local_delta_rescale,
            ) = self._forward_1st_stage(
                original_input=original_input,
                batch=batch,
                local_batch_size=local_batch_size,
                local_max_length=local_max_length,
                rescale=rescale,
                input_mask=input_mask,
                real_lengths=real_lengths,
            )

            # Call to forward pass of the first stage
            loss_2nd_stage = self._forward_2nd_stage(
                local_delta_rescale=local_delta_rescale,
                theta_batch=theta_batch,
                original_max_psd_batch=original_max_psd_batch,
                real_lengths=real_lengths,
            )

            # Total loss
            loss = (
                loss_1st_stage.type(torch.float32)
                + torch.tensor(alpha).to(self.asr_brain.device) * loss_2nd_stage
            )
            loss = torch.mean(loss)

            loss.backward()

            # Do optimization
            self.optimizer_2.step()

            # Save the best adversarial example and adjust the alpha
            # coefficient
            for local_batch_size_idx in range(local_batch_size):
                tokens = (
                    batch.tokens[local_batch_size_idx]
                    .detach()
                    .cpu()
                    .numpy()
                    .reshape(-1)
                )
                pred = np.array(decoded_output[local_batch_size_idx]).reshape(-1)
                if len(pred) == len(tokens) and (pred == tokens).all():
                    if (
                        loss_2nd_stage[local_batch_size_idx]
                        < best_loss_2nd_stage[local_batch_size_idx]
                    ):
                        # Update best loss at 2nd stage
                        best_loss_2nd_stage[local_batch_size_idx] = loss_2nd_stage[
                            local_batch_size_idx
                        ]

                        # Save the best adversarial example
                        successful_adv_input[local_batch_size_idx] = masked_adv_input[
                            local_batch_size_idx
                        ]
                        trans[local_batch_size_idx] = decoded_output[
                            local_batch_size_idx
                        ]

                    # Adjust to increase the alpha coefficient
                    if iter_2nd_stage_idx % self.num_iter_increase_alpha == 0:
                        alpha[local_batch_size_idx] *= self.increase_factor_alpha

                # Adjust to decrease the alpha coefficient
                elif iter_2nd_stage_idx % self.num_iter_decrease_alpha == 0:
                    alpha[local_batch_size_idx] *= self.decrease_factor_alpha
                    alpha[local_batch_size_idx] = max(
                        alpha[local_batch_size_idx], 0.0005
                    )

            # If attack is unsuccessful
            if iter_2nd_stage_idx == self.max_iter_2 - 1:
                for local_batch_size_idx in range(local_batch_size):
                    if successful_adv_input[local_batch_size_idx] is None:
                        successful_adv_input[local_batch_size_idx] = masked_adv_input[
                            local_batch_size_idx
                        ]
                        trans[local_batch_size_idx] = decoded_output[
                            local_batch_size_idx
                        ]

        result = torch.stack(successful_adv_input)  # type: ignore

        return result

    def _forward_2nd_stage(
        self,
        local_delta_rescale: "torch.Tensor",
        theta_batch: List[np.ndarray],
        original_max_psd_batch: List[np.ndarray],
        real_lengths: np.ndarray,
    ):

        # Compute loss for masking threshold
        losses = []
        relu = torch.nn.ReLU()

        for i, _ in enumerate(theta_batch):
            psd_transform_delta = self._psd_transform(
                delta=local_delta_rescale[i, : real_lengths[i]],
                original_max_psd=original_max_psd_batch[i],
            )

            loss = torch.mean(
                relu(psd_transform_delta - theta_batch[i].to(self.asr_brain.device))
            )
            losses.append(loss)

        losses_stack = torch.stack(losses)

        return losses_stack

    def _compute_masking_threshold(
        self, wav: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the masking threshold and the maximum psd of the original audio.
        :param wav: Samples of shape (seq_length,).
        :return: A tuple of the masking threshold and the maximum psd.
        """

        # First compute the psd matrix
        # Get window for the transformation
        # window = scipy.signal.get_window("hann", self.win_length, fftbins=True)
        window = torch.hann_window(self.win_length, periodic=True)
        # Do transformation

        # transformed_wav = librosa.core.stft(
        #    y=x, n_fft=self.n_fft, hop_length=self.hop_length,
        #    win_length=self.win_length, window=window, center=False
        # )
        transformed_wav = torch.stft(
            input=wav.detach().cpu(),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=False,
            return_complex=True,
        ).numpy()
        transformed_wav *= np.sqrt(8.0 / 3.0)

        psd = abs(transformed_wav / self.win_length)
        original_max_psd = np.max(psd * psd)
        with np.errstate(divide="ignore"):
            psd = (20 * np.log10(psd)).clip(min=-200)
        psd = 96 - np.max(psd) + psd

        # Compute freqs and barks
        # freqs = librosa.core.fft_frequencies(
        #    sr=self.asr_brain.hparams.sample_rate, n_fft=self.n_fft)
        freqs = torch.fft.rfftfreq(n=self.n_fft, d=1.0 / 16000)
        barks = 13 * np.arctan(0.00076 * freqs) + 3.5 * np.arctan(
            pow(freqs / 7500.0, 2)
        )

        # Compute quiet threshold
        ath = np.zeros(len(barks), dtype=np.float32) - np.inf
        bark_idx = np.argmax(barks > 1)
        ath[bark_idx:] = (
            3.64 * pow(freqs[bark_idx:] * 0.001, -0.8)
            - 6.5 * np.exp(-0.6 * pow(0.001 * freqs[bark_idx:] - 3.3, 2))
            + 0.001 * pow(0.001 * freqs[bark_idx:], 4)
            - 12
        )

        # Compute the global masking threshold theta
        theta = []

        for i in range(psd.shape[1]):
            # Compute masker index
            masker_idx = argrelextrema(psd[:, i], np.greater)[0]

            if 0 in masker_idx:
                masker_idx = np.delete(masker_idx, 0)

            if len(psd[:, i]) - 1 in masker_idx:
                masker_idx = np.delete(masker_idx, len(psd[:, i]) - 1)

            barks_psd = np.zeros([len(masker_idx), 3], dtype=np.float32)
            barks_psd[:, 0] = barks[masker_idx]
            barks_psd[:, 1] = 10 * np.log10(
                pow(10, psd[:, i][masker_idx - 1] / 10.0)
                + pow(10, psd[:, i][masker_idx] / 10.0)
                + pow(10, psd[:, i][masker_idx + 1] / 10.0)
            )
            barks_psd[:, 2] = masker_idx

            for j in range(len(masker_idx)):
                if barks_psd.shape[0] <= j + 1:
                    break

                while barks_psd[j + 1, 0] - barks_psd[j, 0] < 0.5:
                    quiet_threshold = (
                        3.64 * pow(freqs[int(barks_psd[j, 2])] * 0.001, -0.8)
                        - 6.5
                        * np.exp(
                            -0.6 * pow(0.001 * freqs[int(barks_psd[j, 2])] - 3.3, 2)
                        )
                        + 0.001 * pow(0.001 * freqs[int(barks_psd[j, 2])], 4)
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

            for psd_id in range(barks_psd.shape[0]):
                d_z = barks - barks_psd[psd_id, 0]
                zero_idx = np.argmax(d_z > 0)
                s_f = np.zeros(len(d_z), dtype=np.float32)
                s_f[:zero_idx] = 27 * d_z[:zero_idx]
                s_f[zero_idx:] = (-27 + 0.37 * max(barks_psd[psd_id, 1] - 40, 0)) * d_z[
                    zero_idx:
                ]
                t_s.append(barks_psd[psd_id, 1] + delta[psd_id] + s_f)

            t_s_array = np.array(t_s)

            theta.append(
                np.sum(pow(10, t_s_array / 10.0), axis=0) + pow(10, ath / 10.0)
            )

        theta = np.array(theta)

        return torch.tensor(theta).to(self.asr_brain.device), original_max_psd

    def _psd_transform(
        self, delta: "torch.Tensor", original_max_psd: "torch.Tensor"
    ) -> "torch.Tensor":
        """
        Compute the psd matrix of the perturbation.
        :param delta: The perturbation.
        :param original_max_psd: The maximum psd of the original audio.
        :return: The psd matrix.
        """
        import torch  # lgtm [py/repeated-import]

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
        ).to(self.asr_brain.device)

        # Take abs of complex STFT results
        transformed_delta = torch.sqrt(torch.sum(torch.square(delta_stft), -1))

        # Compute the psd matrix
        psd = (8.0 / 3.0) * transformed_delta / self.win_length
        psd = psd**2
        psd = (
            torch.pow(
                torch.tensor(10.0).type(torch.float32),
                torch.tensor(9.6).type(torch.float32),
            ).to(self.asr_brain.device)
            / torch.reshape(
                torch.tensor(original_max_psd).to(self.asr_brain.device), [-1, 1, 1]
            )
            * psd.type(torch.float32)
        )

        return psd
