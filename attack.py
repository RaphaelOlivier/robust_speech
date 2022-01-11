
import os
import sys
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.pretrained import EncoderDecoderASR
from speechbrain.utils.distributed import run_on_main
from advertorch.attacks import L2PGDAttack
from robust_speech.brains import Seq2SeqASR
from robust_speech.adversarial.attacks.pgd import ASRL2PGDAttack
import torch


models_path= "/home/raphael/dataspace/models/robust_speech/"

hf_repo = "speechbrain"
model_name = "asr-crdnn-rnnlm-librispeech"
source = os.path.join(hf_repo,model_name)
source_hparams = "hyperparams.yaml"
savedir = os.path.join(models_path,model_name)

input_file = "/home/raphael/workspace/libs/speechbrain/samples/audio_samples/example1.wav"

def make_batch(wavform, wrd, tokens,hparams):
    from speechbrain.dataio.batch import PaddedBatch  # noqa

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


if __name__ == "__main__":
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    models_path = "/home/raphael/dataspace/models/robust_speech/"
    asr_model = EncoderDecoderASR.from_hparams(
        source=source, 
        hparams_file= source_hparams, 
        savedir=savedir
    )
    audio_file = input_file
    waveform = asr_model.load_audio(audio_file)
        # Fake a batch:
    wavs = waveform.unsqueeze(0)
    rel_length = torch.tensor([1.0])
    predicted_words, predicted_tokens = asr_model.transcribe_batch(
            wavs, rel_length
        )
    print(predicted_words[0])

    
    import importlib

    with open(hparams_file) as fin:
        brain_hparams = load_hyperpyyaml(fin, overrides)

    run_on_main(brain_hparams["pretrainer"].collect_files)
    brain_hparams["pretrainer"].load_collected(device=run_opts["device"])

    asr_brain = Seq2SeqASR(
        modules=brain_hparams["modules"],
        opt_class=brain_hparams["opt_class"],
        hparams=brain_hparams,
        run_opts=run_opts,
        checkpointer=brain_hparams["checkpointer"],
    )
    asr_brain.tokenizer = brain_hparams["tokenizer"]


    batch = make_batch(waveform,predicted_words[0], predicted_tokens[0], brain_hparams)

    attack = ASRL2PGDAttack(asr_brain, eps=0.05, nb_iter=50, eps_iter=0.005,rand_init=False)
    adv_wavs = attack.perturb(batch)
    print(wavs.norm())
    print((wavs-adv_wavs).norm())
    adv_words, adv_tokens = asr_model.transcribe_batch(
            adv_wavs, rel_length
        )
    print(adv_words[0])

