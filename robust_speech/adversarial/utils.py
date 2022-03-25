"""
Various auxliary functions and classes.
"""

from enum import Enum, auto

import numpy as np
import speechbrain as sb
import torch
import torchaudio
from speechbrain.dataio.batch import PaddedBatch  # noqa
from speechbrain.dataio.preprocess import AudioNormalizer
from speechbrain.pretrained import EncoderDecoderASR
from speechbrain.pretrained.fetching import fetch
from speechbrain.utils.data_utils import split_path


class Stage(Enum):
    """Completes the sb.Stage enum with an attack stage"""

    ATTACK = auto()  # run backward passes through the input
    # predict adversarial example against the attack target (on targeted
    # attacks)
    ADVTARGET = auto()
    ADVTRUTH = auto()  # predict adversarial example against the ground truth


def make_batch_from_waveform(wavform, wrd, tokens, hparams):
    """Make a padded batch from a raw waveform, words and tokens"""
    sig = wavform
    if len(tokens) == 0:  # dummy tokens
        tokens = [3, 4]
    tokens_list = tokens
    tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
    tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
    tokens = torch.LongTensor(tokens_list)
    dic = {
        "id": "0",
        "sig": sig,
        "tokens_list": tokens_list,
        "tokens_bos": tokens_bos,
        "tokens_eos": tokens_eos,
        "tokens": tokens,
        "wrd": wrd,
    }
    return PaddedBatch([dic])


def find_closest_length_string(string, str_list):
    """Find the sentence in str_list whose length is the closest to string"""
    n = len(string)
    lens = [len(s) for s in str_list]
    dist = np.inf
    k = None
    for i, s in enumerate(str_list):
        d = abs(len(s) - n)
        if d < dist:
            dist = d
            k = i
    return str_list[k]


def replace_tokens_in_batch(batch, sent, tokenizer, hparams):
    """Make a padded batch from a raw waveform, words and tokens"""
    assert batch.batchsize == 1, "targeted attacks only support batch size 1"
    if isinstance(sent, list):  # list of possible targets to choose from
        sent = find_closest_length_string(batch.wrd[0], sent)
    if isinstance(tokenizer, sb.dataio.encoder.CTCTextEncoder):
        tokens = tokenizer.encode_sequence(list(sent))
    else:
        tokens = tokenizer.encode_as_ids(sent)

    tokens_list = tokens
    tokens_bos = torch.LongTensor([hparams.bos_index] + (tokens_list))
    tokens_eos = torch.LongTensor(tokens_list + [hparams.eos_index])
    tokens = torch.LongTensor(tokens_list)
    dic = {
        "id": "0",
        "sig": batch.sig[0][0],
        "tokens_list": tokens_list,
        "tokens_bos": tokens_bos,
        "tokens_eos": tokens_eos,
        "tokens": tokens,
    }
    if isinstance(tokenizer, sb.dataio.encoder.CTCTextEncoder):
        dic["char_list"] = list(sent)

    dic["wrd"] = sent
    new_batch = PaddedBatch([dic])
    return new_batch


def transcribe_batch(asr_brain, batch):
    """Outputs transcriptions from an input batch"""
    out = asr_brain.compute_forward(batch, stage=sb.Stage.TEST)
    p_seq, wav_lens, predicted_tokens = out
    try:
        predicted_words = [
            " ".join(asr_brain.tokenizer.decode_ids(utt_seq))
            for utt_seq in predicted_tokens
        ]
    except AttributeError:
        predicted_words = [
            "".join(asr_brain.tokenizer.decode_ndim(utt_seq))
            for utt_seq in predicted_tokens
        ]
    return predicted_words[0], predicted_tokens[0]


def predict_words_from_wavs(hparams, wavs, rel_length):
    """Outputs transcriptions from input wavs tensor"""
    asr_model = EncoderDecoderASR.from_hparams(
        source=hparams["pretrained_model_path"],
        hparams_file=hparams["pretrained_model_hparams_file"],
        savedir=hparams["saved_model_folder"],
    )
    predicted_words, predicted_tokens = asr_model.transcribe_batch(
        wavs, rel_length)
    return predicted_words[0], predicted_tokens[0]


def load_audio(path, hparams, savedir="."):
    """Load a single audio file"""
    source, fl = split_path(path)
    path = fetch(fl, source=source, savedir=savedir)
    signal, sr = torchaudio.load(str(path), channels_first=False)
    audio_normalizer = hparams.get("audio_normalizer", AudioNormalizer())
    return audio_normalizer(signal, sr)


def rand_assign(delta, ord, eps):
    """Randomly set the data of parameter delta with uniform sampling"""
    delta.data.uniform_(-1, 1)
    if ord == np.inf:
        delta.data = eps * delta.data
    elif ord == 2:
        delta.data = l2_clamp_or_normalize(delta.data, eps)
    return delta.data


def l2_clamp_or_normalize(x, eps=None):
    """Clamp x to eps in L2 norm (or normalize if eps is None"""
    xnorm = torch.norm(x, dim=list(range(1, x.dim())))
    if eps:
        coeff = torch.minimum(eps / xnorm, torch.ones_like(xnorm)).unsqueeze(1)
    else:
        coeff = 1.0 / xnorm
    return coeff * x


def linf_clamp(x, eps):
    """Clamp x to eps in Linf norm"""
    if isinstance(eps, torch.Tensor) and eps.dim() == 1:
        eps = eps.unsqueeze(1)
    return torch.clamp(x, min=-eps, max=eps)
