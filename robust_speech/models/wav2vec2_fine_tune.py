"""A Wav2Vec2-based ASR system with librispeech supporting adversarial attacks.
The system employs wav2vec2 as its encoder. Decoding is performed with
ctc greedy decoder.

Inspired from SpeechBrain Wav2Vec2 
(https://github.com/speechbrain/speechbrain/blob/develop/recipes/LibriSpeech/ASR/CTC/train_with_wav2vec.py)
"""

import logging
import os
import sys

import speechbrain as sb
import torch
import torch.nn as nn
from speechbrain.lobes.models.huggingface_wav2vec import HuggingFaceWav2Vec2
from transformers import (
    HubertConfig,
    HubertModel,
    Wav2Vec2Config,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Model,
)

import robust_speech as rs
from robust_speech.adversarial.brain import AdvASRBrain

logger = logging.getLogger(__name__)


# Define training procedure


class W2VASR(AdvASRBrain):
    """
    Wav2Vec 2.0 ASR model
    """

    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        if not stage == rs.Stage.ATTACK:
            batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        # if self.filter is not None:
        #     wavs = self.filter(wavs)
        tokens_bos, _ = batch.tokens_bos
        # wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
        # Add augmentation if specified

        if hasattr(self.hparams, "smoothing") and self.hparams.smoothing:
            wavs = self.hparams.smoothing(wavs, wav_lens)

        if stage == sb.Stage.TRAIN:
            if hasattr(self.modules, "env_corrupt"):
                wavs_noise = self.modules.env_corrupt(wavs, wav_lens)
                wavs = torch.cat([wavs, wavs_noise], dim=0)
                wav_lens = torch.cat([wav_lens, wav_lens])
                tokens_bos = torch.cat([tokens_bos, tokens_bos], dim=0)

            if hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs, wav_lens)
        # Forward pass
        if stage == rs.Stage.ATTACK:
            feats = self.modules.wav2vec2.extract_features(wavs)
            encoded = self.modules.enc(feats)
        else:
            feats = self.modules.wav2vec2(wavs)
            encoded = self.modules.enc(feats.detach())
        # Compute outputs
        p_tokens = None
        logits = self.modules.ctc_lin(encoded)
        p_ctc = self.hparams.log_softmax(logits)

        if stage not in [sb.Stage.TRAIN, rs.Stage.ATTACK]:
            p_tokens = sb.decoders.ctc_greedy_decode(
                p_ctc, wav_lens, blank_id=self.hparams.blank_index
            )
        return p_ctc, wav_lens, p_tokens

    def compute_objectives(
        self, predictions, batch, stage, adv=False, targeted=False, reduction="mean"
    ):
        """Computes the loss (CTC+NLL) given predictions and targets."""

        p_ctc, wav_lens, predicted_tokens = predictions

        ids = batch.id
        tokens_eos, tokens_eos_lens = batch.tokens_eos
        tokens, tokens_lens = batch.tokens

        if hasattr(self.modules, "env_corrupt") and stage == sb.Stage.TRAIN:
            tokens_eos = torch.cat([tokens_eos, tokens_eos], dim=0)
            tokens_eos_lens = torch.cat(
                [tokens_eos_lens, tokens_eos_lens], dim=0)
            tokens = torch.cat([tokens, tokens], dim=0)
            tokens_lens = torch.cat([tokens_lens, tokens_lens], dim=0)

        loss_ctc = self.hparams.ctc_cost(
            p_ctc, tokens, wav_lens, tokens_lens, reduction=reduction
        )
        loss = loss_ctc

        if stage != sb.Stage.TRAIN and stage != rs.Stage.ATTACK:
            # Decode token terms to words
            if isinstance(self.tokenizer, sb.dataio.encoder.CTCTextEncoder):
                predicted_words = [
                    self.tokenizer.decode_ndim(utt_seq) for utt_seq in predicted_tokens
                ]
                predicted_words = ["".join(s).strip().split(" ")
                                   for s in predicted_words]
            else:
                predicted_words = [
                    self.tokenizer.decode_ids(utt_seq).split(" ") for utt_seq in predicted_tokens
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

        return loss

    def on_stage_end(
        self, stage, stage_loss, epoch, stage_adv_loss=None, stage_adv_loss_target=None
    ):
        """Gets called at the end of an epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage_adv_loss is not None:
            stage_stats["adv loss"] = stage_adv_loss
        if stage_adv_loss_target is not None:
            stage_stats["adv loss target"] = stage_adv_loss_target
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")
            if stage_adv_loss is not None:
                stage_stats["adv CER"] = self.adv_cer_metric.summarize(
                    "error_rate")
                stage_stats["adv WER"] = self.adv_wer_metric.summarize(
                    "error_rate")
            if stage_adv_loss_target is not None:
                stage_stats["adv CER target"] = self.adv_cer_metric_target.summarize(
                    "error_rate"
                )
                stage_stats["adv WER target"] = self.adv_wer_metric_target.summarize(
                    "error_rate"
                )

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            current_lr = self.hparams.noam_annealing.current_lr
            steps = self.hparams.noam_annealing.n_steps
            self.hparams.train_logger.log_stats(
                stats_meta={
                    "epoch": epoch,
                    "lr_model": current_lr,
                },
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"WER": stage_stats["WER"]},
                min_keys=["WER"],
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Evaluation stage": "TEST"},
                test_stats=stage_stats,
            )
            with open(self.hparams.wer_file, "w") as wer:
                self.wer_metric.write_stats(wer)

    def init_optimizers(self):
        "Initializes the wav2vec2 optimizer and model optimizer"
        self.optimizer = self.hparams.opt_class(
            self.hparams.model.parameters()
        )

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable("optimizer", self.optimizer)

    def fit_batch(self, batch):
        """Train the parameters given a single batch in input"""
        # check if we need to switch optimizer
        # if so change the optimizer from Adam to SGD

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
