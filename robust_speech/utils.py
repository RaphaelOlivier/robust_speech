from enum import Enum, auto
import torch
import torchaudio
import speechbrain as sb
from speechbrain.dataio.batch import PaddedBatch  # noqa
from speechbrain.utils.data_utils import split_path
from speechbrain.pretrained.fetching import fetch
from speechbrain.dataio.preprocess import AudioNormalizer
from speechbrain.pretrained import EncoderDecoderASR

class Stage(Enum):
    """Completes the sb.Stage enum with an attack stage"""

    ATTACK=auto()

def make_batch_from_waveform(wavform, wrd, tokens,hparams):

    sig = wavform
    tokens_list = tokens
    tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
    tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
    tokens = torch.LongTensor(tokens_list)
    dic = {
        "id":"0",
        "sig":sig,
        "tokens_list":tokens_list,
        "tokens_bos":tokens_bos,
        "tokens_eos":tokens_eos,
        "tokens":tokens,
        "wrd":wrd
    }

    return PaddedBatch([dic])

def transcribe_batch(asr_brain, batch):
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
    asr_model = EncoderDecoderASR.from_hparams(
        source=hparams["pretrained_model_path"], 
        hparams_file= hparams["pretrained_model_hparams_file"], 
        savedir=hparams["saved_model_folder"]
    )
    predicted_words, predicted_tokens = asr_model.transcribe_batch(
        wavs, rel_length
    )
    return predicted_words[0], predicted_tokens[0]

def load_audio(path, hparams, savedir="."):
    source, fl = split_path(path)
    path = fetch(fl, source=source, savedir=savedir)
    signal, sr = torchaudio.load(str(path), channels_first=False)
    audio_normalizer = hparams.get(
        "audio_normalizer", AudioNormalizer()
    )
    return audio_normalizer(signal, sr)