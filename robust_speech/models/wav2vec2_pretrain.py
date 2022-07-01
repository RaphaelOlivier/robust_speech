"""A Wav2Vec2 Pretraining system with librispeech supporting adversarial attacks, 
and specifically the contrastive attack.
The HuggingFace implementation of the wav2vec 2.0 pretraining is used and wrapped
to fit properly the SpeechBrain framework.

Contrary to ASR models this one requires some additional work over SpeechBrain
(https://github.com/speechbrain/speechbrain/blob/develop/recipes/CommonVoice/self-supervised-learning/wav2vec2/train.py)
in order to:
    -support loading of pretrained models from Huggingface 
    (Speechbrain handles it for Wav2Vec2 for ASR but not pretraining)
    -support the quantized_representation argument to fix the quantized labels
     used by Wav2Vec2 (required for the contrastive attack).
    -backpropagate gradients to the inputs
Some transformers and SpeechBrain models have been rewritten below for that purpose.
"""
import logging
import sys

import numpy as np
import speechbrain as sb
import torch
import torch.nn.functional as F
import transformers
from speechbrain.lobes.models.huggingface_wav2vec import HuggingFaceWav2Vec2Pretrain
from transformers import Wav2Vec2ForPreTraining
from transformers.file_utils import (
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.models.wav2vec2.configuration_wav2vec2 import Wav2Vec2Config
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    _CONFIG_FOR_DOC,
    WAV_2_VEC_2_INPUTS_DOCSTRING,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForPreTrainingOutput,
    _compute_mask_indices,
)

import robust_speech as rs
from robust_speech.adversarial.brain import AdvASRBrain

logger = logging.getLogger(__name__)


# Define training procedure


class W2VPretrain(AdvASRBrain):
    """
    Wav2Vec 2.0 base model for pretraining
    """

    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the w2v2 loss."""
        wavs, wav_lens = batch.sig

        if not stage == rs.Stage.ATTACK:
            batch = batch.to(self.device)
            wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)

        if hasattr(self.hparams, "smoothing") and self.hparams.smoothing:
            wavs = self.hparams.smoothing(wavs, wav_lens)
        # Forward on w2v2 and take the loss.
        # It has to be on train mode even for eval. Otherwise it would deactivate
        # the loss computation ...
        # used saved quantized representation (prior to attack)
        if hasattr(batch, "quantized_representation"):
            out, mask = self.modules.wav2vec2(
                wavs, quantized_representation=batch.quantized_representation
            )
        else:
            # compute quantized representation on the fly
            out, mask = self.modules.wav2vec2(
                wavs, quantized_representation=None)

        if stage == rs.Stage.ATTACK:
            loss = out.contrastive_loss
        else:
            loss = out.loss

        if stage != sb.Stage.TRAIN and stage != rs.Stage.ATTACK:
            return loss, out, mask
        return loss

    def compute_objectives(
        self, predictions, batch, stage, adv=False, targeted=False, reduction="mean"
    ):
        """Computes the loss (CTC+NLL) given predictions and targets."""
        if stage == sb.Stage.TRAIN or stage == rs.Stage.ATTACK:
            # We don't have to compute anything as the HF model directly returns
            # the constrative loss.
            loss = predictions
        else:
            # We compute the accuracy between embeddings with cosing sim.
            loss, out, mask_time_indices = predictions
            cosine_sim = torch.cosine_similarity(
                out.projected_states, out.projected_quantized_states, dim=-1
            )
            # acc = cosine_sim[mask_time_indices].mean()
            acc = (
                torch.masked_select(cosine_sim, mask_time_indices.bool())
                .mean()
                .detach()
            )
            if adv:
                if targeted:
                    self.adv_acc_metric_target.append(acc)
                else:
                    self.adv_acc_metric.append(acc)
            else:
                self.acc_metric.append(acc)
        return loss

    def fit_batch(self, batch):
        """Train the parameters given a single batch in input"""

        # Here we manage mixed precision
        if self.auto_mix_prec:
            with torch.cuda.amp.autocast():
                predictions = self.compute_forward(batch, sb.Stage.TRAIN)
                loss = self.compute_objectives(
                    predictions, batch, sb.Stage.TRAIN)

            # normalize the loss by gradient_accumulation step
            self.scaler.scale(
                loss / self.hparams.gradient_accumulation).backward()

            if self.step % self.hparams.gradient_accumulation == 0:
                # gradient clipping & early stop if loss is not fini
                self.check_gradients(loss)

                self.scaler.unscale_(self.optimizer)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                # anneal lr every update
                self.hparams.noam_annealing(self.optimizer)
        else:
            predictions = self.compute_forward(batch, sb.Stage.TRAIN)
            loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)

            # normalize the loss by gradient_accumulation step
            (loss / self.hparams.gradient_accumulation).backward()

            if self.step % self.hparams.gradient_accumulation == 0:
                # gradient clipping & early stop if loss is not fini
                self.check_gradients(loss)

                self.optimizer.step()
                self.optimizer.zero_grad()

                # anneal lr every update
                self.hparams.noam_annealing(self.optimizer)

        return loss.detach()

    def fit_batch_adversarial(self, batch):
        """Train the parameters given a single batch in input"""

        # Here we manage mixed precision
        if self.auto_mix_prec:
            with torch.cuda.amp.autocast():
                predictions, _ = self.compute_forward_adversarial(
                    batch, sb.Stage.TRAIN)
                loss = self.compute_objectives(
                    predictions, batch, sb.Stage.TRAIN)

            # normalize the loss by gradient_accumulation step
            self.scaler.scale(
                loss / self.hparams.gradient_accumulation).backward()

            if self.step % self.hparams.gradient_accumulation == 0:
                # gradient clipping & early stop if loss is not fini
                self.check_gradients(loss)

                self.scaler.unscale_(self.optimizer)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                # anneal lr every update
                self.hparams.noam_annealing(self.optimizer)
        else:
            predictions = self.compute_forward(batch, sb.Stage.TRAIN)
            loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)

            # normalize the loss by gradient_accumulation step
            (loss / self.hparams.gradient_accumulation).backward()

            if self.step % self.hparams.gradient_accumulation == 0:
                # gradient clipping & early stop if loss is not fini
                self.check_gradients(loss)

                self.optimizer.step()
                self.optimizer.zero_grad()

                # anneal lr every update
                self.hparams.noam_annealing(self.optimizer)

        return loss.detach()

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage == sb.Stage.VALID or stage == sb.Stage.TEST:
            self.acc_metric = []
            self.adv_acc_metric = []
            self.adv_acc_metric_target = []

    def on_stage_end(
        self, stage, stage_loss, epoch, stage_adv_loss=None, stage_adv_loss_target=None
    ):
        num_to_keep = self.hparams.num_to_keep if "num_to_keep" in self.hparams.__dict__ else 1
        """Gets called at the end of an epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage_adv_loss is not None:
            stage_stats["adv_loss"] = stage_adv_loss
        if stage_adv_loss_target is not None:
            stage_stats["adv_loss target"] = stage_adv_loss_target
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["acc"] = sum(self.acc_metric) / len(self.acc_metric)
            if stage_adv_loss is not None:
                stage_stats["adv acc"] = sum(self.adv_acc_metric) / len(
                    self.adv_acc_metric
                )
            if stage_adv_loss_target is not None:
                stage_stats["adv acc target"] = sum(self.adv_acc_metric_target) / len(
                    self.adv_acc_metric_target
                )

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            current_lr = self.hparams.noam_annealing.current_lr
            steps = self.hparams.noam_annealing.n_steps
            optimizer = self.optimizer.__class__.__name__

            epoch_stats = {
                "epoch": epoch,
                "lr": current_lr,
                "steps": steps,
                "optimizer": optimizer,
            }
            self.hparams.train_logger.log_stats(
                stats_meta=epoch_stats,
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"acc": stage_stats["acc"], "epoch": epoch},
                max_keys=["acc"],
                num_to_keep=num_to_keep,
            )

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Evaluation stage": "TEST"},
                test_stats=stage_stats,
            )
