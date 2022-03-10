#!/usr/bin/env/python3
"""A RNN-Transducer ASR system with librispeech supporting adversarial attacks.
The system employs an encoder, a decoder, and an joint network
between them. Decoding is performed with beamsearch coupled with a neural
language model.

Inspired from SpeechBrain Transducer (https://github.com/speechbrain/speechbrain/blob/develop/recipes/LibriSpeech/ASR/transducer/train.py)
"""
import os
import sys
import torch
import logging
import speechbrain as sb
from speechbrain.utils.distributed import run_on_main
from hyperpyyaml import load_hyperpyyaml

import robust_speech as rs
from robust_speech.adversarial.brain import AdvASRBrain

logger = logging.getLogger(__name__)

# Define training procedure

class RNNTASR(AdvASRBrain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        if not stage == rs.Stage.ATTACK:
            batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        tokens_with_bos, token_with_bos_lens = batch.tokens_bos
        # wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)

        # Add augmentation if specified
        if stage == sb.Stage.TRAIN:
            if hasattr(self.modules, "env_corrupt"):
                wavs_noise = self.modules.env_corrupt(wavs, wav_lens)
                wavs = torch.cat([wavs, wavs_noise], dim=0)
                wav_lens = torch.cat([wav_lens, wav_lens])
                batch.sig = wavs, wav_lens
                tokens_with_bos = torch.cat(
                    [tokens_with_bos, tokens_with_bos], dim=0
                )
                token_with_bos_lens = torch.cat(
                    [token_with_bos_lens, token_with_bos_lens]
                )
                batch.tokens_bos = tokens_with_bos, token_with_bos_lens
            if hasattr(self.modules, "augmentation"):
                wavs = self.modules.augmentation(wavs, wav_lens)

        # Forward pass
        feats = self.hparams.compute_features(wavs)
        if stage == sb.Stage.TRAIN:
            feats = self.modules.normalize(feats, wav_lens)
        else:
            feats = self.modules.normalize(feats, wav_lens,epoch=self.modules.normalize.update_until_epoch+1) # don't update normalization outside of training!
        if stage == rs.Stage.ATTACK:
            x = self.modules.enc(feats)
        else:
            x = self.modules.enc(feats.detach())
        e_in = self.modules.emb(tokens_with_bos)
        h, _ = self.modules.dec(e_in)
        # Joint network
        # add labelseq_dim to the encoder tensor: [B,T,H_enc] => [B,T,1,H_enc]
        # add timeseq_dim to the decoder tensor: [B,U,H_dec] => [B,1,U,H_dec]
        joint = self.modules.Tjoint(x.unsqueeze(2), h.unsqueeze(1))

        # Output layer for transducer log-probabilities
        logits = self.modules.transducer_lin(joint)
        p_transducer = self.hparams.log_softmax(logits)

        # Compute outputs
        if stage == sb.Stage.TRAIN or stage == rs.Stage.ATTACK:
            return_CTC = False
            return_CE = False
            current_epoch = self.hparams.epoch_counter.current
            if (
                hasattr(self.hparams, "ctc_cost")
                and current_epoch <= self.hparams.number_of_ctc_epochs
            ):
                return_CTC = True
                # Output layer for ctc log-probabilities
                out_ctc = self.modules.enc_lin(x)
                p_ctc = self.hparams.log_softmax(out_ctc)
            if (
                hasattr(self.hparams, "ce_cost")
                and current_epoch <= self.hparams.number_of_ce_epochs
            ):
                return_CE = True
                # Output layer for ctc log-probabilities
                p_ce = self.modules.dec_lin(h)
                p_ce = self.hparams.log_softmax(p_ce)
            if return_CE and return_CTC:
                return p_ctc, p_ce, p_transducer, wav_lens
            elif return_CTC:
                return p_ctc, p_transducer, wav_lens
            elif return_CE:
                return p_ce, p_transducer, wav_lens
            else:
                return p_transducer, wav_lens

        elif stage == sb.Stage.VALID:
            best_hyps, scores, _, _ = self.hparams.Greedysearcher(x)
            return p_transducer, wav_lens, best_hyps
        else:
            (
                best_hyps,
                best_scores,
                nbest_hyps,
                nbest_scores,
            ) = self.hparams.Beamsearcher(x)
            return p_transducer, wav_lens, best_hyps

    def compute_objectives(self, predictions, batch, stage, adv = False):
        """Computes the loss (Transducer+(CTC+NLL)) given predictions and targets."""

        ids = batch.id
        current_epoch = self.hparams.epoch_counter.current
        tokens, token_lens = batch.tokens
        tokens_eos, token_eos_lens = batch.tokens_eos
        if hasattr(self.modules, "env_corrupt") and stage == sb.Stage.TRAIN:
            tokens_eos = torch.cat([tokens_eos, tokens_eos], dim=0)
            token_eos_lens = torch.cat([token_eos_lens, token_eos_lens], dim=0)
            tokens = torch.cat([tokens, tokens], dim=0)
            token_lens = torch.cat([token_lens, token_lens], dim=0)

        if stage == sb.Stage.TRAIN or stage == rs.Stage.ATTACK:
            if len(predictions) == 4:
                p_ctc, p_ce, p_transducer, wav_lens = predictions
                CTC_loss = self.hparams.ctc_cost(
                    p_ctc, tokens, wav_lens, token_lens
                )
                CE_loss = self.hparams.ce_cost(
                    p_ce, tokens_eos, length=token_eos_lens
                )
                loss_transducer = self.hparams.transducer_cost(
                    p_transducer, tokens, wav_lens, token_lens
                )
                loss = (
                    self.hparams.ctc_weight * CTC_loss
                    + self.hparams.ce_weight * CE_loss
                    + (1 - (self.hparams.ctc_weight + self.hparams.ce_weight))
                    * loss_transducer
                )
            elif len(predictions) == 3:
                # one of the 2 heads (CTC or CE) is still computed
                # CTC alive
                if current_epoch <= self.hparams.number_of_ctc_epochs:
                    p_ctc, p_transducer, wav_lens = predictions
                    CTC_loss = self.hparams.ctc_cost(
                        p_ctc, tokens, wav_lens, token_lens
                    )
                    loss_transducer = self.hparams.transducer_cost(
                        p_transducer, tokens, wav_lens, token_lens
                    )
                    loss = (
                        self.hparams.ctc_weight * CTC_loss
                        + (1 - self.hparams.ctc_weight) * loss_transducer
                    )
                # CE for decoder alive
                else:
                    p_ce, p_transducer, wav_lens = predictions
                    CE_loss = self.hparams.ce_cost(
                        p_ce, tokens_eos, length=token_eos_lens
                    )
                    loss_transducer = self.hparams.transducer_cost(
                        p_transducer, tokens, wav_lens, token_lens
                    )
                    loss = (
                        self.hparams.ce_weight * CE_loss
                        + (1 - self.hparams.ctc_weight) * loss_transducer
                    )
            else:
                p_transducer, wav_lens = predictions
                loss = self.hparams.transducer_cost(
                    p_transducer, tokens, wav_lens, token_lens
                )
        else:
            p_transducer, wav_lens, predicted_tokens = predictions
            loss = self.hparams.transducer_cost(
                p_transducer, tokens, wav_lens, token_lens
            )

        if stage not in [sb.Stage.TRAIN, rs.Stage.ATTACK]:
            # Decode token terms to words
            predicted_words = [
                self.tokenizer.decode_ids(utt_seq).split(" ")
                for utt_seq in predicted_tokens
            ]
            target_words = [wrd.split(" ") for wrd in batch.wrd]
            if adv:
                self.adv_wer_metric.append(ids, predicted_words, target_words)
                self.adv_cer_metric.append(ids, predicted_words, target_words)
            else:
                self.wer_metric.append(ids, predicted_words, target_words)
                self.cer_metric.append(ids, predicted_words, target_words)

        return loss
