#!/usr/bin/env python3

import sys
import torch
import logging
import speechbrain as sb

from robust_speech.adversarial.brain import AdvASRBrain
import robust_speech as rs

from transformers import (
    Wav2Vec2ForPreTraining, 
    Wav2Vec2ForPreTrainingOutput, 
    WAV_2_VEC_2_INPUTS_DOCSTRING,
    _CONFIG_FOR_DOC
)
from transformers.file_utils import (
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)

from speechbrain.lobes.huggingface_wav2vec import HuggingFaceWav2Vec2Pretrain
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

class AdvWav2Vec2ForPreTraining(Wav2Vec2ForPreTraining):
    @add_start_docstrings_to_model_forward(WAV_2_VEC_2_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Wav2Vec2ForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_values,
        attention_mask=None,
        mask_time_indices=None,
        sampled_negative_indices=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        quantized_representation=None,
    ):
        r"""
        mask_time_indices (`torch.BoolTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices to mask extracted features for contrastive loss. When in training mode, model learns to predict
            masked extracted features in *config.proj_codevector_dim* space.
        sampled_negative_indices (`torch.BoolTensor` of shape `(batch_size, sequence_length, num_negatives)`, *optional*):
            Indices indicating which quantized target vectors are used as negative sampled vectors in contrastive loss.
            Required input for pre-training.
        Returns:
        Example:
        ```python
        >>> import torch
        >>> from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForPreTraining
        >>> from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices
        >>> from datasets import load_dataset
        >>> import soundfile as sf
        >>> feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("patrickvonplaten/wav2vec2-base")
        >>> model = Wav2Vec2ForPreTraining.from_pretrained("patrickvonplaten/wav2vec2-base")
        >>> def map_to_array(batch):
        ...     speech, _ = sf.read(batch["file"])
        ...     batch["speech"] = speech
        ...     return batch
        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> ds = ds.map(map_to_array)
        >>> input_values = feature_extractor(ds["speech"][0], return_tensors="pt").input_values  # Batch size 1
        >>> # compute masked indices
        >>> batch_size, raw_sequence_length = input_values.shape
        >>> sequence_length = model._get_feat_extract_output_lengths(raw_sequence_length)
        >>> mask_time_indices = _compute_mask_indices((batch_size, sequence_length), mask_prob=0.2, mask_length=2)
        >>> mask_time_indices = torch.tensor(mask_time_indices, device=input_values.device, dtype=torch.long)
        >>> with torch.no_grad():
        ...     outputs = model(input_values, mask_time_indices=mask_time_indices)
        >>> # compute cosine similarity between predicted (=projected_states) and target (=projected_quantized_states)
        >>> cosine_sim = torch.cosine_similarity(outputs.projected_states, outputs.projected_quantized_states, dim=-1)
        >>> # show that cosine similarity is much higher than random
        >>> cosine_sim[mask_time_indices.to(torch.bool)].mean() > 0.5
        tensor(True)
        >>> # for contrastive loss training model should be put into train mode
        >>> model = model.train()
        >>> loss = model(input_values, mask_time_indices=mask_time_indices).loss
        ```"""

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if mask_time_indices is not None:
            mask_time_indices = mask_time_indices.to(torch.bool)

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            mask_time_indices=mask_time_indices,
            return_dict=return_dict,
        )

        # 1. project all transformed features (including masked) to final vq dim
        transformer_features = self.project_hid(outputs[0])

        # 2. quantize all (unmasked) extracted features and project to final vq dim
        extract_features = self.dropout_features(outputs[1])

        if attention_mask is not None:
            # compute reduced attention_mask correponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask, add_adapter=False
            )
        if quantized_representation is not None:
            quantized_features, codevector_perplexity = quantized_representation
        else:
            quantized_features, codevector_perplexity = self.quantizer(
                extract_features, mask_time_indices=mask_time_indices
            )
            quantized_features = self.project_q(quantized_features)

        loss = contrastive_loss = diversity_loss = None
        if sampled_negative_indices is not None:
            batch_size, sequence_length, hidden_size = quantized_features.shape

            # for training, we sample negatives
            # 3. sample K negatives (distractors) quantized states for contrastive loss
            # if attention_mask is passed, make sure that padded feature vectors cannot be sampled
            # sample negative quantized vectors BTC => (BxT)C
            negative_quantized_features = quantized_features.view(-1, hidden_size)[
                sampled_negative_indices.long().view(-1)
            ]
            negative_quantized_features = negative_quantized_features.view(
                batch_size, sequence_length, -1, hidden_size
            ).permute(2, 0, 1, 3)

            # 4. compute logits, corresponding to `logs = sim(c_t, [q_t, \sim{q}_t]) / \kappa`
            # of equation (3) in https://arxiv.org/pdf/2006.11477.pdf
            logits = self.compute_contrastive_logits(
                quantized_features[None, :],
                negative_quantized_features,
                transformer_features,
                self.config.contrastive_logits_temperature,
            )

            # 5. if a negative vector is identical to the positive (i.e. when codebook utilization is low),
            # its cosine similarity will be masked
            neg_is_pos = (quantized_features == negative_quantized_features).all(-1)

            if neg_is_pos.any():
                logits[1:][neg_is_pos] = float("-inf")

            # 6. compute contrastive loss \mathbf{L}_m = cross_entropy(logs) =
            # -log(exp(sim(c_t, q_t)/\kappa) / \sum_{\sim{q}} exp(sim(c_t, \sim{q})/\kappa))
            logits = logits.transpose(0, 2).reshape(-1, logits.size(0))
            target = ((1 - mask_time_indices.long()) * -100).transpose(0, 1).flatten()

            contrastive_loss = torch.nn.functional.cross_entropy(logits.float(), target, reduction="sum")
            # 7. compute diversity loss: \mathbf{L}_d
            num_codevectors = self.config.num_codevectors_per_group * self.config.num_codevector_groups
            diversity_loss = ((num_codevectors - codevector_perplexity) / num_codevectors) * mask_time_indices.sum()

            # 8. \mathbf{L} = \mathbf{L}_m + \alpha * \mathbf{L}_d
            loss = contrastive_loss + self.config.diversity_loss_weight * diversity_loss

        if not return_dict:
            if loss is not None:
                return (loss, transformer_features, quantized_features, codevector_perplexity) + outputs[2:]
            return (transformer_features, quantized_features, codevector_perplexity) + outputs[2:]

        return Wav2Vec2ForPreTrainingOutput(
            loss=loss,
            projected_states=transformer_features,
            projected_quantized_states=quantized_features,
            codevector_perplexity=codevector_perplexity,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            contrastive_loss=contrastive_loss,
            diversity_loss=diversity_loss,
        )

class AdvHuggingFaceWav2Vec2Pretrain(HuggingFaceWav2Vec2Pretrain):
    def __init__(
        self,
        source,
        save_path,
        mask_prob=0.65,
        mask_length=10,
        normalize_wav=True,
    ):
        super(AdvHuggingFaceWav2Vec2Pretrain,self).__init__(
            source,
            save_path,
            mask_prob=0.65,
            mask_length=10,
            normalize_wav=True
        )
        self.model = AdvWav2Vec2ForPreTraining(self.config)
        self.model.gradient_checkpointing_disable()  # Required by DDP
        self.model.train()

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
        if stage == rs.Stage.ATTACK and hasattr(batch, "quantized_representation"):
            out, mask = self.modules.wav2vec2(
                wavs, 
                quantized_representation=batch.quantized_representation
            )
        else:
            out, mask = self.modules.wav2vec2(wavs)

        if stage == rs.Stage.ATTACK:
            loss = out.contrastive_loss
        else:
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
        if stage == sb.Stage.VALID or stage == sb.Stage.TEST:
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
