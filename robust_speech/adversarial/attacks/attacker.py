"""
An abstract attack class and a simple baseline attack.
"""

import numpy as np
import torch
import torch.nn as nn

from robust_speech.adversarial.metrics import AudioSaver, SNRComputer
from robust_speech.adversarial.utils import rand_assign


class Attacker:
    """
    Abstract class for running attacks and logging results (SNR and audio files)

    Arguments
    ---------
    asr_brain: rs.adversarial.brain.ASRBrain
        brain object.
    targeted: bool
        if the attack is targeted.
    """

    def on_evaluation_start(self, load_audio=False, save_audio_path=None, sample_rate=16000):
        """
        Method to run at the beginning of an evaluation phase with adverersarial attacks.

        Arguments
        ---------
        save_audio_path: optional string
            path to the folder in which to save audio files
        sample_rate: int
            audio sample rate for wav encoding
        """

        self.load_audio = load_audio
        if self.load_audio and (save_audio_path is None):
            raise ValueError(
                "save_audio_path must be provided in order to load audio files")
        self.snr_metric = SNRComputer()
        self.save_audio_path = save_audio_path
        if self.save_audio_path:
            self.audio_saver = AudioSaver(save_audio_path, sample_rate)

    def on_evaluation_end(self, logger):
        """
        Method to run at the end of an evaluation phase with adverersarial attacks.

        Arguments
        ---------
        logger: sb.utils,train_logger.FileLogger
            path to the folder in which to save audio files
        """
        snr = self.snr_metric.summarize()
        snr = {
            "average": snr["average"],
            "min_score": snr["min_score"],
            "max_score": snr["max_score"],
        }
        logger.log_stats(
            stats_meta={},
            test_stats={"Adversarial SNR": snr},
        )

    def on_fit_start(self):
        pass

    def on_fit_end(self):
        pass

    def fit(self, loader):
        """
        Fitting the parameters of the attacker, if applicable (generative attack, universal attack, etc)
        Arguments
        ---------
        loader:
            data loader to use to fit the attacker
        """
        raise NotImplementedError

    def perturb_and_log(self, batch, target=None):
        """
        Compute an adversarial perturbation and log results

        Arguments
        ---------
        batch : sb.PaddedBatch
            The input batch to perturb

        Returns
        -------
        the tensor of the perturbed batch
        """
        adv_wav = None
        if self.load_audio:
            adv_wav = self.audio_saver.load(batch.id, batch)
        if adv_wav is None:
            adv_wav = self.perturb(batch)
            if self.save_audio_path:
                self.audio_saver.save(batch.id, batch, adv_wav)
        self.snr_metric.append(batch.id, batch, adv_wav)
        return adv_wav

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

        raise NotImplementedError


class RandomAttack(Attacker):
    """
    An attack based on entirely random noise,
    which can be used as a baseline with various noise bounds.
    The attack returns a noisy input within eps from the initial point.

    Arguments
    ---------
     asr_brain: rs.adversarial.brain.ASRBrain
        brain object.
     eps: float
        maximum distortion.
     clip_min: (optional) float
        mininum value per input dimension.
     clip_max: (optional) float
        maximum value per input dimension.
     order: (optional) int
         the order of maximum distortion (inf or 2).
     targeted: bool
        if the attack is targeted (not used).
    """

    def __init__(
        self,
        asr_brain,
        eps=0.3,
        order=np.inf,
        clip_min=None,
        clip_max=None,
        targeted=False,
    ):
        self.asr_brain = asr_brain
        self.eps = eps
        self.order = order
        self.clip_min = clip_min
        self.clip_max = clip_max
        assert not targeted

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

        save_input = batch.sig[0]
        wav_init = torch.clone(save_input)
        delta = torch.zeros_like(wav_init)
        delta = nn.Parameter(delta)
        clip_min = self.clip_min if self.clip_min is not None else -10
        clip_max = self.clip_max if self.clip_max is not None else 10
        rand_assign(delta, self.order, self.eps)
        delta.data = (
            torch.clamp(wav_init + delta.data, min=clip_min,
                        max=clip_max) - wav_init
        )
        wav_adv = wav_init + delta.data
        return wav_adv
