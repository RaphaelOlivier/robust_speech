"""A Transformer ASR system with librispeech supporting adversarial attacks.
The system employs an encoder, a decoder, and an attention mechanism
between them. Decoding is performed with (CTC/Att joint) beamsearch coupled with a neural
language model.

Inspired from SpeechBrain Transformer (https://github.com/speechbrain/speechbrain/blob/develop/recipes/LibriSpeech/ASR/transformer/train.py)
"""
import logging
import os
import sys

import speechbrain as sb
import torch

import robust_speech as rs
from robust_speech.adversarial.brain import AdvASRBrain

logger = logging.getLogger(__name__)


# Define training procedure
class TrfASR(AdvASRBrain):
    """
    Encoder-decoder models using Transformers
    """

    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        if not stage == rs.Stage.ATTACK:
            batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        tokens_bos, _ = batch.tokens_bos

        # Add augmentation if specified
        if stage == sb.Stage.TRAIN:
            if hasattr(self.modules, "env_corrupt"):
                wavs_noise = self.modules.env_corrupt(wavs, wav_lens)
                wavs = torch.cat([wavs, wavs_noise], dim=0)
                wav_lens = torch.cat([wav_lens, wav_lens])
                tokens_bos = torch.cat([tokens_bos, tokens_bos], dim=0)

        # compute features
        feats = self.hparams.compute_features(wavs)
        current_epoch = self.hparams.epoch_counter.current
        feats = self.modules.normalize(feats, wav_lens, epoch=current_epoch)

        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "augmentation"):
                feats = self.hparams.augmentation(feats)

        # forward modules
        src = self.modules.CNN(feats)
        enc_out, pred = self.modules.Transformer(
            src, tokens_bos, wav_lens, pad_idx=self.hparams.pad_index
        )

        p_ctc = None
        if self.hparams.ctc_weight > 0.0:
            # output layer for ctc log-probabilities
            logits = self.modules.ctc_lin(enc_out)
            p_ctc = self.hparams.log_softmax(logits)

        p_seq = None
        # if self.hparams.ctc_weight<1.:
        # output layer for seq2seq log-probabilities
        pred = self.modules.seq_lin(pred)
        p_seq = self.hparams.log_softmax(pred)
        # Compute outputs
        hyps = None
        if stage == sb.Stage.TRAIN or stage == rs.Stage.ATTACK:
            hyps = None
        elif stage == sb.Stage.VALID:
            hyps = None
            current_epoch = self.hparams.epoch_counter.current
            if current_epoch % self.hparams.valid_search_interval == 0:
                # for the sake of efficiency, we only perform beamsearch with limited capacity
                # and no LM to give user some idea of how the AM is doing
                hyps, _ = self.hparams.valid_search(enc_out.detach(), wav_lens)
        else:
            hyps, _ = self.hparams.test_search(enc_out.detach(), wav_lens)

        return p_ctc, p_seq, wav_lens, hyps

    def compute_objectives(
        self, predictions, batch, stage, adv=False, targeted=False, reduction="mean"
    ):
        """Computes the loss (CTC+NLL) given predictions and targets."""

        (
            p_ctc,
            p_seq,
            wav_lens,
            hyps,
        ) = predictions

        ids = batch.id
        tokens_eos, tokens_eos_lens = batch.tokens_eos
        tokens, tokens_lens = batch.tokens

        if hasattr(self.modules, "env_corrupt") and stage == sb.Stage.TRAIN:
            tokens_eos = torch.cat([tokens_eos, tokens_eos], dim=0)
            tokens_eos_lens = torch.cat(
                [tokens_eos_lens, tokens_eos_lens], dim=0)
            tokens = torch.cat([tokens, tokens], dim=0)
            tokens_lens = torch.cat([tokens_lens, tokens_lens], dim=0)
        loss_seq = 0.0
        if self.hparams.ctc_weight < 1.0:
            loss_seq = self.hparams.seq_cost(
                p_seq, tokens_eos, length=tokens_eos_lens, reduction=reduction
            )
        loss_ctc = 0.0
        if self.hparams.ctc_weight > 0.0:
            loss_ctc = self.hparams.ctc_cost(
                p_ctc, tokens, wav_lens, tokens_lens, reduction=reduction
            )
        loss = (
            self.hparams.ctc_weight * loss_ctc
            + (1 - self.hparams.ctc_weight) * loss_seq
        )

        if stage != sb.Stage.TRAIN and stage != rs.Stage.ATTACK:
            current_epoch = self.hparams.epoch_counter.current
            valid_search_interval = self.hparams.valid_search_interval
            if current_epoch % valid_search_interval == 0 or (stage == sb.Stage.TEST):
                # Decode token terms to words
                predicted_words = [
                    self.tokenizer.decode_ids(utt_seq).split(" ") for utt_seq in hyps
                ]
                target_words = [wrd.split(" ") for wrd in batch.wrd]
                if adv:
                    if targeted:
                        self.adv_wer_metric_target.append(
                            ids, predicted_words, target_words
                        )
                        self.adv_cer_metric_target.append(
                            ids, predicted_words, target_words
                        )
                    else:
                        self.adv_wer_metric.append(
                            ids, predicted_words, target_words)
                        self.adv_cer_metric.append(
                            ids, predicted_words, target_words)
                else:
                    self.wer_metric.append(ids, predicted_words, target_words)
                    self.cer_metric.append(ids, predicted_words, target_words)

            # compute the accuracy of the one-step-forward prediction
            if adv:
                if targeted:
                    self.adv_acc_metric_target.append(
                        p_seq, tokens_eos, tokens_eos_lens
                    )
                else:
                    self.adv_acc_metric.append(
                        p_seq, tokens_eos, tokens_eos_lens)
            else:
                self.acc_metric.append(p_seq, tokens_eos, tokens_eos_lens)

        return loss

    def fit_batch(self, batch):
        """Train the parameters given a single batch in input"""
        # check if we need to switch optimizer
        # if so change the optimizer from Adam to SGD
        self.check_and_reset_optimizer()

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

    def check_and_reset_optimizer(self):
        """reset the optimizer if training enters stage 2"""
        current_epoch = self.hparams.epoch_counter.current
        if not hasattr(self, "switched"):
            self.switched = False
            if isinstance(self.optimizer, torch.optim.SGD):
                self.switched = True

        if self.switched is True:
            return

        if current_epoch > self.hparams.stage_one_epochs:
            self.optimizer = self.hparams.SGD(self.modules.parameters())

            if self.checkpointer is not None:
                self.checkpointer.add_recoverable("optimizer", self.optimizer)

            self.switched = True

    def fit_batch_adversarial(self, batch):
        """Train the parameters given a single batch in input"""
        # check if we need to switch optimizer
        # if so change the optimizer from Adam to SGD
        self.check_and_reset_optimizer()

        predictions, _ = self.compute_forward_adversarial(
            batch, sb.Stage.TRAIN)
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
        # Gets called at the beginning of each epoch
        if stage != sb.Stage.TRAIN:
            self.acc_metric = self.hparams.acc_computer()
            self.wer_metric = self.hparams.error_rate_computer()
            self.cer_metric = self.hparams.cer_computer()
            self.adv_acc_metric = self.hparams.acc_computer()
            self.adv_cer_metric = self.hparams.cer_computer()
            self.adv_wer_metric = self.hparams.error_rate_computer()
            self.adv_acc_metric_target = self.hparams.acc_computer()
            self.adv_cer_metric_target = self.hparams.cer_computer()
            self.adv_wer_metric_target = self.hparams.error_rate_computer()

    def on_stage_end(
        self, stage, stage_loss, epoch, stage_adv_loss=None, stage_adv_loss_target=None
    ):
        # Gets called at the end of a epoch.
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage_adv_loss is not None:
            stage_stats["adv loss"] = stage_adv_loss
        if stage_adv_loss_target is not None:
            stage_stats["adv loss target"] = stage_adv_loss_target
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["ACC"] = self.acc_metric.summarize()
            if stage_adv_loss is not None:
                stage_stats["adv ACC"] = self.adv_acc_metric.summarize()
            if stage_adv_loss_target is not None:
                stage_stats["adv ACC target"] = self.adv_acc_metric_target.summarize()
            current_epoch = self.hparams.epoch_counter.current
            valid_search_interval = self.hparams.valid_search_interval
            if current_epoch % valid_search_interval == 0 or stage == sb.Stage.TEST:
                stage_stats["WER"] = self.wer_metric.summarize("error_rate")
                stage_stats["CER"] = self.cer_metric.summarize("error_rate")

                if stage_adv_loss is not None:
                    stage_stats["adv CER"] = self.adv_cer_metric.summarize(
                        "error_rate")
                    stage_stats["adv WER"] = self.adv_wer_metric.summarize(
                        "error_rate")
                if stage_adv_loss_target is not None:
                    stage_stats[
                        "adv CER target"
                    ] = self.adv_cer_metric_target.summarize("error_rate")
                    stage_stats[
                        "adv WER target"
                    ] = self.adv_wer_metric_target.summarize("error_rate")

        # log stats and save checkpoint at end-of-epoch
        if stage == sb.Stage.VALID and sb.utils.distributed.if_main_process():

            # report different epoch stages according current stage
            current_epoch = self.hparams.epoch_counter.current
            if current_epoch <= self.hparams.stage_one_epochs:
                lr = self.hparams.noam_annealing.current_lr
                steps = self.hparams.noam_annealing.n_steps
                optimizer = self.optimizer.__class__.__name__
            else:
                lr = self.hparams.lr_sgd
                steps = -1
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
            if self.checkpointer is not None:
                self.checkpointer.save_and_keep_only(
                    meta={"ACC": stage_stats["ACC"], "epoch": epoch},
                    max_keys=["ACC"],
                    num_to_keep=5,
                )

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={
                    "Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            with open(self.hparams.wer_file, "w") as w:
                self.wer_metric.write_stats(w)

            # save the averaged checkpoint at the end of the evaluation stage
            # delete the rest of the intermediate checkpoints
            # ACC is set to 1.1 so checkpointer only keeps the averaged
            # checkpoint
            if self.checkpointer is not None:
                self.checkpointer.save_and_keep_only(
                    meta={"ACC": 1.1, "epoch": epoch},
                    max_keys=["ACC"],
                    num_to_keep=1,
                )
