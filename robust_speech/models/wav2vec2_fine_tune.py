"""A Wav2Vec2-based ASR system with librispeech supporting adversarial attacks.
The system employs wav2vec2 as its encoder. Decoding is performed with
ctc greedy decoder.

Inspired from SpeechBrain Wav2Vec2 (https://github.com/speechbrain/speechbrain/blob/develop/recipes/LibriSpeech/ASR/CTC/train_with_wav2vec.py)
"""

import gc
import logging
import os
import sys

import speechbrain as sb
import torch
import torch.nn as nn
from speechbrain.lobes.models.huggingface_wav2vec import HuggingFaceWav2Vec2
from transformers import (HubertConfig, HubertModel, Wav2Vec2Config,
                          Wav2Vec2FeatureExtractor, Wav2Vec2Model)

import robust_speech as rs
from robust_speech.adversarial.brain import AdvASRBrain
from robust_speech.models.wav2vec2_pretrain import AdvWav2Vec2FeatureEncoder

logger = logging.getLogger(__name__)


class AdvWav2Vec2Model(Wav2Vec2Model):
    """
    This class modifies the transformers Wav2Vec2 module in order to replace the Feature Extractor with AdvWav2Vec2FeatureEncoder
    """

    def __init__(self, config: Wav2Vec2Config):
        super().__init__(config)
        self.feature_extractor = AdvWav2Vec2FeatureEncoder(config)


HF_models = {"wav2vec2": AdvWav2Vec2Model, "hubert": HubertModel}
HF_config = {"wav2vec2": Wav2Vec2Config, "hubert": HubertConfig}


class AdvHuggingFaceWav2Vec2(HuggingFaceWav2Vec2):
    """This class inherits the SpeechBrain Wav2Vec2 lobe and replaces the model with an AdvWav2Vec2 model,
    which supports backpropagating through the inputs

    Arguments
    ---------
    source : str
        HuggingFace hub name: e.g "facebook/wav2vec2-large-lv60"
    save_path : str
        Path (dir) of the downloaded model.
    output_norm : bool (default: True)
        If True, a layer_norm (affine) will be applied to the output obtained
        from the wav2vec model.
    freeze : bool (default: True)
        If True, the model is frozen. If False, the model will be trained
        alongside with the rest of the pipeline.
    freeze_feature_extractor :  bool (default: False)
        When freeze = False and freeze_feature_extractor True, the featue_extractor module of the model is Frozen. If False
        all the wav2vec model will be trained including featue_extractor module.
    apply_spec_augment : bool (default: False)
        If True, the model will apply spec augment on the output of feature extractor
        (inside huggingface Wav2VecModel() class).
        If False, the model will not apply spec augment. We set this to false to prevent from doing it twice.
    """

    def __init__(
        self,
        source,
        save_path,
        output_norm=True,
        freeze=True,
        freeze_feature_extractor=False,
        apply_spec_augment=False,
    ):
        nn.Module.__init__(self)

        # Download the extractor from HuggingFace.
        # The extractor is only used to retrieve the normalisation information
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            source, cache_dir=save_path
        )

        # Select specific self-supervised loader (eg. Wav2Vec2, Hubert)
        if "hubert" in source:
            config = HF_config.get("hubert")
            model = HF_models.get("hubert")
        else:
            config = HF_config.get("wav2vec2")
            model = HF_models.get("wav2vec2")

        # Download and load the model
        self._from_pretrained(source, config=config,
                              model=model, save_path=save_path)

        # set apply_spec_augment
        self.model.config.apply_spec_augment = apply_spec_augment

        # We check if inputs need to be normalized w.r.t pretrained wav2vec2
        self.normalize_wav = self.feature_extractor.do_normalize

        self.freeze = freeze
        self.freeze_feature_extractor = freeze_feature_extractor
        self.output_norm = output_norm
        if self.freeze:
            logger.warning(
                "speechbrain.lobes.models.huggingface_wav2vec - wav2vec 2.0 is frozen."
            )
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            self.model.train()
            if self.freeze_feature_extractor:
                self.model.feature_extractor._freeze_parameters()


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
        tokens_bos, _ = batch.tokens_bos
        # wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
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
        if stage == rs.Stage.ATTACK:
            feats = self.modules.wav2vec2.extract_features(wavs)
            x = self.modules.enc(feats)
        else:
            feats = self.modules.wav2vec2(wavs)
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
            predicted_words = [
                self.tokenizer.decode_ndim(utt_seq) for utt_seq in predicted_tokens
            ]
            target_words = [wrd for wrd in batch.wrd]
            predicted_words = ["".join(s).strip().split(" ")
                               for s in predicted_words]
            target_words = [t.split(" ") for t in target_words]
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

    def fit_batch(self, batch):
        """Train the parameters given a single batch in input"""
        predictions = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)
        loss.backward()
        if self.check_gradients(loss):
            self.wav2vec_optimizer.step()
            self.model_optimizer.step()

        self.wav2vec_optimizer.zero_grad()
        self.model_optimizer.zero_grad()

        return loss.detach().cpu()

    def fit_batch_adversarial(self, batch):
        """Train the parameters given a single batch in input"""
        predictions, _ = self.compute_forward_adversarial(
            batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)
        loss.backward()
        if self.check_gradients(loss):
            self.wav2vec_optimizer.step()
            self.model_optimizer.step()

        self.wav2vec_optimizer.zero_grad()
        self.model_optimizer.zero_grad()

        return loss.detach().cpu()

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
            old_lr_model, new_lr_model = self.hparams.lr_annealing_model(
                stage_stats["loss"]
            )
            old_lr_wav2vec, new_lr_wav2vec = self.hparams.lr_annealing_wav2vec(
                stage_stats["loss"]
            )
            sb.nnet.schedulers.update_learning_rate(
                self.model_optimizer, new_lr_model)
            sb.nnet.schedulers.update_learning_rate(
                self.wav2vec_optimizer, new_lr_wav2vec
            )
            self.hparams.train_logger.log_stats(
                stats_meta={
                    "epoch": epoch,
                    "lr_model": old_lr_model,
                    "lr_wav2vec": old_lr_wav2vec,
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
                stats_meta={
                    "Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            with open(self.hparams.wer_file, "w") as w:
                self.wer_metric.write_stats(w)

    def init_optimizers(self):
        "Initializes the wav2vec2 optimizer and model optimizer"
        self.wav2vec_optimizer = self.hparams.wav2vec_opt_class(
            self.modules.wav2vec2.parameters()
        )
        self.model_optimizer = self.hparams.model_opt_class(
            self.hparams.model.parameters()
        )

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable(
                "wav2vec_opt", self.wav2vec_optimizer)
            self.checkpointer.add_recoverable("modelopt", self.model_optimizer)
