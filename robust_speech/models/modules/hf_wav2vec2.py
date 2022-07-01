import logging
import os
import sys
from typing import Iterable
import numpy as np
import torch.nn.functional as F
import speechbrain as sb
import torch
import torch.nn as nn
from speechbrain.lobes.models.huggingface_wav2vec import HuggingFaceWav2Vec2
from transformers import (
    HubertConfig,
    HubertModel,
    Wav2Vec2Config,
    Wav2Vec2ConformerConfig,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Model,
    Wav2Vec2ConformerModel,
    Data2VecAudioConfig,
    Data2VecAudioModel,
)
from transformers.models.data2vec.modeling_data2vec_audio import Data2VecAudioFeatureEncoder
from speechbrain.pretrained.fetching import fetch
import transformers
from speechbrain.lobes.models.huggingface_wav2vec import HuggingFaceWav2Vec2Pretrain, HF_models, HF_config
from transformers import Wav2Vec2ForPreTraining
from transformers.file_utils import (
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.models.wav2vec2.configuration_wav2vec2 import Wav2Vec2Config as Wav2Vec2PretrainConfig
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    _CONFIG_FOR_DOC,
    WAV_2_VEC_2_INPUTS_DOCSTRING,
    Wav2Vec2ForPreTrainingOutput,
    _compute_mask_indices,
)
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2FeatureExtractor as Wav2Vec2PretrainFeatureExtractor
import robust_speech as rs

logger = logging.getLogger(__name__)


class AdvWav2Vec2FeatureEncoder(Wav2Vec2PretrainFeatureExtractor):
    """
    Slight modification of the HF feature extractor.
    The original class assumes that input is a leaf tensor,
    which when running attacks isn't always the case.
    """

    def forward(self, input_values):
        hidden_states = input_values[:, None]
        # make sure hidden_states require grad for gradient_checkpointing
        if (
            self._requires_grad and self.training and hidden_states.is_leaf
        ):  # not always true when attacking
            hidden_states.requires_grad = True

        for conv_layer in self.conv_layers:
            if self._requires_grad and self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(conv_layer),
                    hidden_states,
                )
            else:
                hidden_states = conv_layer(hidden_states)

        return hidden_states


class AdvData2VecAudioFeatureEncoder(Data2VecAudioFeatureEncoder):
    def forward(self, input_values):
        hidden_states = input_values[:, None]

        # make sure hidden_states require grad for gradient_checkpointing
        if self._requires_grad and self.training and hidden_states.is_leaf:
            hidden_states.requires_grad = True

        for conv_layer in self.conv_layers:
            if self._requires_grad and self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(conv_layer),
                    hidden_states,
                )
            else:
                hidden_states = conv_layer(hidden_states)

        return hidden_states


class AdvWav2Vec2ForPreTraining(Wav2Vec2ForPreTraining):
    """
    This class modifies the transformers Wav2Vec2ForPreTraining module in order to
        -replace the Feature Extractor with AdvWav2Vec2FeatureEncoder
        -handle contrastive attacks in forward
    """

    def __init__(self, config: Wav2Vec2PretrainConfig):
        super().__init__(config)
        self.wav2vec2.feature_extractor = AdvWav2Vec2FeatureEncoder(config)

    @add_start_docstrings_to_model_forward(WAV_2_VEC_2_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=Wav2Vec2ForPreTrainingOutput, config_class=_CONFIG_FOR_DOC
    )
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
        """
        New argument quantized_representation contains an optional
        precomputed value for (quantized_features, codevector_perplexity).
        If available, this value is not recomputed in the foward pass.

        Returns:
        --------
        """

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

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

        # 1. project all transformed features (including masked) to final vq
        # dim
        transformer_features = self.project_hid(outputs[0])

        # 2. quantize all (unmasked) extracted features and project to final vq
        # dim
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
            # if attention_mask is passed,
            # make sure that padded feature vectors cannot be sampled
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

            # 5. if a negative vector is identical to the positive
            # (i.e. when codebook utilization is low),
            # its cosine similarity will be masked
            neg_is_pos = (quantized_features ==
                          negative_quantized_features).all(-1)

            if neg_is_pos.any():
                logits[1:][neg_is_pos] = float("-inf")

            # 6. compute contrastive loss \mathbf{L}_m = cross_entropy(logs) =
            # -log(exp(sim(c_t, q_t)/\kappa) / \sum_{\sim{q}} exp(sim(c_t, \sim{q})/\kappa))
            logits = logits.transpose(0, 2).reshape(-1, logits.size(0))
            target = ((1 - mask_time_indices.long())
                      * -100).transpose(0, 1).flatten()

            contrastive_loss = torch.nn.functional.cross_entropy(
                logits.float(), target, reduction="sum"
            )
            # 7. compute diversity loss: \mathbf{L}_d
            num_codevectors = (
                self.config.num_codevectors_per_group
                * self.config.num_codevector_groups
            )
            diversity_loss = (
                (num_codevectors - codevector_perplexity) / num_codevectors
            ) * mask_time_indices.sum()

            # 8. \mathbf{L} = \mathbf{L}_m + \alpha * \mathbf{L}_d
            loss = contrastive_loss + self.config.diversity_loss_weight * diversity_loss
        if not return_dict:
            if loss is not None:
                return (
                    loss,
                    transformer_features,
                    quantized_features,
                    codevector_perplexity,
                ) + outputs[2:]
            return (
                transformer_features,
                quantized_features,
                codevector_perplexity,
            ) + outputs[2:]

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
    """This lobe enables the integration of HuggingFace
     wav2vec2.0 models to be pretrained.
     It also enables contrastive attacks and parameter loading from HuggingFace.

    Arguments
    ---------
    source : str
        HuggingFace hub name: e.g "facebook/wav2vec2-large-lv60"
    save_path : str
        Path (dir) of the downloaded model.
    mask_prob : float (default: 0.65)
        Probability of masking a given frame. Default is taken from the paper.
    mask_length : float (default: 10)
        Length (i.e. number of consecutive masked frames). Default is taken from
        the paper.
    """

    def __init__(
        self,
        source,
        save_path,
        mask_prob=0.65,
        mask_length=10,
        normalize_wav=True,
        load_pretrained_weights=False
    ):
        super(AdvHuggingFaceWav2Vec2Pretrain, self).__init__(
            source,
            save_path,
            mask_prob=mask_prob,
            mask_length=mask_length,
            normalize_wav=normalize_wav,
        )
        if load_pretrained_weights:
            self.model = AdvWav2Vec2ForPreTraining.from_pretrained(
                source, load_weight=load_pretrained_weights)
        else:
            self.config = Wav2Vec2PretrainConfig.from_pretrained(
                source, cache_dir=save_path
            )
            self.config.output_hidden_states = (
                True  # We want the hidden states as well!
            )

            self.model = AdvWav2Vec2ForPreTraining(self.config)

    def forward(self, wav, quantized_representation=None):
        """Takes an input waveform and return its corresponding wav2vec encoding.

        Arguments
        ---------
        wav : torch.Tensor (signal)
            A batch of audio signals to transform to features.
        quantized_representation : Optional[torch.Tensor,torch.Tensor]
            A precomputed quantized representation of the audio signal.
        """
        batch_size, raw_sequence_length = wav.shape
        if self.normalize_wav:
            wav = F.layer_norm(wav, wav.shape)
        sequence_length = self.model._get_feat_extract_output_lengths(
            raw_sequence_length
        )

        # 1. Compute the indices that will be masked
        mask_time_indices = _compute_mask_indices(
            (batch_size, sequence_length),
            mask_prob=self.mask_prob,
            mask_length=self.mask_length,
        )
        torch_mask_time_indices = torch.tensor(
            mask_time_indices,
            device=wav.device,
            dtype=torch.long,
        )

        # 2. Sample the negative samples from the entire sequence.
        # Fairseq does it only on the masked indices, but this only work if you
        # have long sentences. For more versatily, we sample on the entire sequence.
        # value.
        full_sentence_indices = np.ones((batch_size, sequence_length))
        # print(np.sum(mask_time_indices, axis=1))
        negative_sample_indices = torch.tensor(
            transformers.models.wav2vec2.modeling_wav2vec2._sample_negative_indices(
                (batch_size, sequence_length.numpy()),
                num_negatives=self.config.num_negatives,
                mask_time_indices=full_sentence_indices,
            ),
            device=wav.device,
            dtype=torch.long,
        )
        return (
            self.model(
                wav,
                mask_time_indices=torch_mask_time_indices,
                sampled_negative_indices=negative_sample_indices,
                quantized_representation=quantized_representation,
            ),
            torch_mask_time_indices,
        )


class AdvWav2Vec2Model(Wav2Vec2Model):
    """
    This class modifies the transformers Wav2Vec2 module
     in order to replace the Feature Extractor with AdvWav2Vec2FeatureEncoder
    """

    def __init__(self, config: Wav2Vec2Config):
        super().__init__(config)
        self.feature_extractor = AdvWav2Vec2FeatureEncoder(config)


class AdvWav2Vec2ConformerModel(Wav2Vec2ConformerModel):
    """
    This class modifies the transformers Wav2Vec2 module
     in order to replace the Feature Extractor with AdvWav2Vec2FeatureEncoder
    """

    def __init__(self, config: Wav2Vec2ConformerConfig):
        super().__init__(config)
        self.feature_extractor = AdvWav2Vec2FeatureEncoder(config)


class AdvHubertModel(HubertModel):
    """
    This class modifies the transformers Wav2Vec2 module
     in order to replace the Feature Extractor with AdvWav2Vec2FeatureEncoder
    """

    def __init__(self, config: HubertConfig):
        super().__init__(config)
        self.feature_extractor = AdvWav2Vec2FeatureEncoder(config)


class AdvData2VecAudioModel(Data2VecAudioModel):
    """
    This class modifies the transformers Data2VecAudio module
     in order to replace the Feature Extractor with AdvData2VecAudioFeatureEncoder
    """

    def __init__(self, config: Data2VecAudioConfig):
        super().__init__(config)
        self.feature_extractor = AdvData2VecAudioFeatureEncoder(config)


Adv_HF_models = {
    "wav2vec2": AdvWav2Vec2Model,
    "wav2vec2-conformer": AdvWav2Vec2ConformerModel,
    "hubert": AdvHubertModel,
    "data2vec": AdvData2VecAudioModel
}


Adv_HF_config = {
    "wav2vec2": Wav2Vec2Config,
    "hubert": HubertConfig,
    "wav2vec2-conformer": Wav2Vec2ConformerConfig,
    "data2vec": Data2VecAudioConfig
}


class AdvHuggingFaceWav2Vec2(HuggingFaceWav2Vec2):
    """This class inherits the SpeechBrain Wav2Vec2 lobe and
    replaces the model with an AdvWav2Vec2 model,
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
        When freeze = False and freeze_feature_extractor True,
        the featue_extractor module of the model is Frozen. If False
        all the wav2vec model will be trained including featue_extractor module.
    apply_spec_augment : bool (default: False)
        If True, the model will apply spec augment on the output of feature extractor
        (inside huggingface Wav2VecModel() class).
        If False, the model will not apply spec augment.
        We set this to false to prevent from doing it twice.
    """

    def __init__(
        self,
        source,
        save_path,
        output_norm=True,
        freeze=True,
        freeze_feature_extractor=False,
        apply_spec_augment=False,
        load_pretrained_weights=True,
        dropout=None,
    ):
        nn.Module.__init__(self)
        # Download the extractor from HuggingFace.
        # The extractor is only used to retrieve the normalisation information
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            source, cache_dir=save_path
        )

        # Select specific self-supervised loader (eg. Wav2Vec2, Hubert)
        if "hubert" in source:
            config = Adv_HF_config.get("hubert")
            model = Adv_HF_models.get("hubert")
        elif "data2vec" in source:
            config = Adv_HF_config.get("data2vec")
            model = Adv_HF_models.get("data2vec")
        elif "wav2vec2-conformer" in source:
            config = Adv_HF_config.get("wav2vec2-conformer")
            model = Adv_HF_models.get("wav2vec2-conformer")
        else:
            config = Adv_HF_config.get("wav2vec2")
            model = Adv_HF_models.get("wav2vec2")

        # Download and load the model
        self._from_pretrained(source, config=config, model=model,
                              save_path=save_path, load_weights=load_pretrained_weights, dropout=dropout)

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

    def _from_pretrained(self, source, config, model, save_path, load_weights, dropout=None):
        """This function manages the source checking and loading of the params.
        # 1. Is the model from HF or a local path
        # 2. Is the model pretrained with HF or SpeechBrain
        # 3. Download (if appropriate) and load with respect to 1. and 2.
        """

        def override_dropout(config, dropout):
            if isinstance(config, dict):
                for key in config:
                    if isinstance(config[key], dict):
                        override_dropout(config[key], dropout)
                    elif isinstance(key, str) and "dropout" in key and config[key] > 0:
                        config[key] = dropout
            elif isinstance(config, list):
                return
            else:
                override_dropout(config.__dict__, dropout)
        config = config.from_pretrained(source, cache_dir=save_path)
        if dropout:
            override_dropout(config, dropout)

        if not load_weights:
            self.model = model(config)
            return
        else:
            is_sb, ckpt_file = self._check_model_source(source)
            if is_sb:
                self.model = model(config)
                self.model.gradient_checkpointing_disable()  # Required by DDP
                # fetch the checkpoint file
                ckpt_full_path = fetch(
                    filename=ckpt_file, source=source, savedir=save_path
                )
                # We transfer the parameters from the checkpoint.
                self._load_sb_pretrained_w2v2_parameters(ckpt_full_path)
            else:
                self.model = model.from_pretrained(
                    source, config=config, cache_dir=save_path)

    def extract_features(self, wav):
        """Takes an input waveform and return its corresponding wav2vec encoding.

        Arguments
        ---------
        wav : torch.Tensor (signal)
            A batch of audio signals to transform to features.
        """

        if self.normalize_wav:
            wav = F.layer_norm(wav, wav.shape)

        # Extract wav2vec output
        out = self.model(wav)[0]
        # We normalize the output if required
        if self.output_norm:
            out = F.layer_norm(out, out.shape)
        return out
