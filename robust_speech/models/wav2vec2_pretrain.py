#!/usr/bin/env python3

import sys
import torch
import logging
import speechbrain as sb

from robust_speech.adversarial.brain import AdvASRBrain
import robust_speech as rs
"""Recipe for pretraining a wav2vec 2.0 model on CommonVoice EN. Note that it can be
trained with ANY dataset as long as you provide the correct JSON or CSV file.

The HuggingFace implementation of the wav2vec 2.0 pretraining is used and wrapped
to fit properly the SpeechBrain framework. Models have been compared to the original
fairseq implementation with success. The Transformers HuggingFace library is
required:
> pip install extra_requirements.txt

Hence the process is the following:
1. Indicate a HuggingFace repository that stores the wav2vec 2.0 config file.
This is necessary to determine the architecture of the model that will be
instantiated.
2. Train it with our wrapper.
3. Save it to be reused as a pretrained encoder within SpeechBrain (or others).

wav2vec 2.0: https://arxiv.org/abs/2006.11477
HuggingFace: https://huggingface.co/transformers/model_doc/wav2vec2.html

To run this recipe, do the following:
> python train.py hparams/hyperparams.yaml


Authors
 * Titouan Parcollet 2021
 * Yan Gao 2021
"""

logger = logging.getLogger(__name__)


# Define training procedure
class W2VPretrain(AdvASRBrain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the w2v2 loss."""

        if not stage == rs.Stage.ATTACK:
            batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)

        # Forward on w2v2 and take the loss.
        # It has to be on train mode even for eval. Otherwise it would deactivate
        # the loss computation ...
        out, mask = self.modules.wav2vec2(wavs)
        loss = out.loss

        if stage != sb.Stage.TRAIN and stage != rs.Stage.ATTACK:
            return loss, out, mask

        return loss

    def compute_objectives(self, predictions, batch, stage, adv = False):
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
            acc = cosine_sim[mask_time_indices].mean()
            if adv:
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
                    predictions, batch, sb.Stage.TRAIN
                )

            # normalize the loss by gradient_accumulation step
            self.scaler.scale(
                loss / self.hparams.gradient_accumulation
            ).backward()

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
                predictions = self.compute_forward_adversarial(batch, sb.Stage.TRAIN)
                loss = self.compute_objectives(
                    predictions, batch, sb.Stage.TRAIN
                )

            # normalize the loss by gradient_accumulation step
            self.scaler.scale(
                loss / self.hparams.gradient_accumulation
            ).backward()

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
        if stage != sb.Stage.TRAIN:
            self.acc_metric = []
            self.adv_acc_metric = []

    def on_stage_end(self, stage, stage_loss, epoch, stage_adv_loss=None):
        """Gets called at the end of an epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage_adv_loss is not None:
            stage_stats["adv_loss"] = stage_adv_loss
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["acc"] = sum(self.acc_metric) / len(self.acc_metric)
            if stage_adv_loss is not None:
                stage_stats["adv acc"] = sum(self.adv_acc_metric) / len(self.adv_acc_metric)

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            lr = self.hparams.noam_annealing.current_lr
            steps = self.hparams.noam_annealing.n_steps
            optimizer = self.optimizer.__class__.__name__

            epoch_stats = {
                "epoch": epoch,
                "lr": lr,
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
            )

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
