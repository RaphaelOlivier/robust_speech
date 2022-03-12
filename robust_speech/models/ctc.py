#!/usr/bin/env/python3
"""A CTC ASR system with librispeech supporting adversarial attacks.
The system can employ any encoder. Decoding is performed with
ctc greedy decoder.
The neural network is trained on CTC likelihood target and character units
are used as basic recognition tokens. Training is performed on the full
LibriSpeech dataset (960 h).

Inspired from both SpeechBrain Wav2Vec2 
(https://github.com/speechbrain/speechbrain/blob/develop/recipes/LibriSpeech/ASR/CTC/train_with_wav2vec.py)
and Seq2Seq (https://github.com/speechbrain/speechbrain/blob/develop/recipes/LibriSpeech/ASR/seq2seq/train.py)

"""

import torch
import logging
import speechbrain as sb
from robust_speech.adversarial.brain import AdvASRBrain
import robust_speech as rs

logger = logging.getLogger(__name__)


# Define training procedure
class CTCASR(AdvASRBrain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        if not stage == rs.Stage.ATTACK:
            batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        tokens_bos, _ = batch.tokens_bos
        #wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
        # Add augmentation if specified
        if stage == sb.Stage.TRAIN:
            if hasattr(self.modules, "env_corrupt"):
                wavs_noise = self.modules.env_corrupt(wavs, wav_lens)
                wavs = torch.cat([wavs, wavs_noise], dim=0)
                wav_lens = torch.cat([wav_lens, wav_lens])
                tokens_bos = torch.cat([tokens_bos, tokens_bos], dim=0)

            if hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs, wav_lens)
        # Forward pass
        feats = self.hparams.compute_features(wavs)
        if stage == sb.Stage.TRAIN:
            feats = self.modules.normalize(feats, wav_lens)
        else:
            # don't update normalization outside of training!
            feats = self.modules.normalize(
                feats, wav_lens, epoch=self.modules.normalize.update_until_epoch+1)
        if stage == rs.Stage.ATTACK:
            x = self.modules.enc(feats)
        else:
            x = self.modules.enc(feats.detach())
         # Compute outputs
        p_tokens = None
        logits = self.modules.ctc_lin(x)
        p_ctc = self.hparams.log_softmax(logits)

        if stage not in [sb.Stage.TRAIN, rs.Stage.ATTACK]:
            p_tokens = sb.decoders.ctc_greedy_decode(
                p_ctc, wav_lens, blank_id=self.hparams.blank_index
            )
        return p_ctc, wav_lens, p_tokens

    def compute_objectives(self, predictions, batch, stage, adv=False, reduction="mean"):
        """Computes the loss (CTC+NLL) given predictions and targets."""

        p_ctc, wav_lens, predicted_tokens = predictions

        ids = batch.id
        tokens_eos, tokens_eos_lens = batch.tokens_eos
        tokens, tokens_lens = batch.tokens

        if hasattr(self.modules, "env_corrupt") and stage == sb.Stage.TRAIN:
            tokens_eos = torch.cat([tokens_eos, tokens_eos], dim=0)
            tokens_eos_lens = torch.cat(
                [tokens_eos_lens, tokens_eos_lens], dim=0
            )
            tokens = torch.cat([tokens, tokens], dim=0)
            tokens_lens = torch.cat([tokens_lens, tokens_lens], dim=0)

        loss_ctc = self.hparams.ctc_cost(
            p_ctc, tokens, wav_lens, tokens_lens, reduction=reduction)
        loss = loss_ctc

        if stage != sb.Stage.TRAIN and stage != rs.Stage.ATTACK:
            # Decode token terms to words
            predicted_words = [
                self.tokenizer.decode_ndim(utt_seq)
                for utt_seq in predicted_tokens
            ]
            target_words = [wrd for wrd in batch.wrd]
            if adv:
                self.adv_wer_metric.append(ids, predicted_words, target_words)
                self.adv_cer_metric.append(ids, predicted_words, target_words)
            else:
                self.wer_metric.append(ids, predicted_words, target_words)
                self.cer_metric.append(ids, predicted_words, target_words)

        return loss


class CTCGreedyDecode(sb.decoders.seq2seq.S2SBaseSearcher):
    """Binding between Seq2Seq models and CTC decoders"""

    def __init__(self, blank_index, ctc_lin, log_softmax):
        super(CTCGreedyDecode, self).__init__(None, None, None, None)
        self.blank_index = blank_index
        self.ctc_lin = ctc_lin
        self.log_softmax = log_softmax

    def forward(self, enc_states, wav_len):
        logits = self.ctc_lin(enc_states)
        p_ctc = self.log_softmax(logits)
        p_tokens = sb.decoders.ctc_greedy_decode(
            p_ctc, wav_len, blank_id=self.blank_index
        )
        return p_tokens, None
