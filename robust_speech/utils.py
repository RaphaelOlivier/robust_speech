import torch
import torchaudio
from speechbrain.dataio.batch import PaddedBatch  # noqa
from speechbrain.utils.data_utils import split_path
from speechbrain.pretrained.fetching import fetch
from speechbrain.dataio.preprocess import AudioNormalizer
from speechbrain.pretrained import EncoderDecoderASR

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