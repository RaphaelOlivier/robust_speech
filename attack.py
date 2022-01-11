
import os
import sys
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.pretrained import EncoderDecoderASR
import foolbox as fb
from robust_speech.adversarial.wrapper import BrainModel, BrainMisclassification
from robust_speech.brains import Seq2SeqASR
import torch


models_path= "/home/raphael/dataspace/models/robust_speech/"

hf_repo = "speechbrain"
model_name = "asr-crdnn-rnnlm-librispeech"
source = os.path.join(hf_repo,model_name)
source_hparams = "hyperparams.yaml"
savedir = os.path.join(models_path,model_name)
train_hparams = "recipes/natural/seq2seq/hparams/train.yaml"

input_file = "/home/raphael/workspace/libs/speechbrain/samples/audio_samples/example1.wav"


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
    batch = waveform.unsqueeze(0)
    rel_length = torch.tensor([1.0])
    predicted_words, predicted_tokens = asr_model.transcribe_batch(
            batch, rel_length
        )
    print(batch.min(),batch.max())
    print(predicted_words[0])

    import importlib

    with open(train_hparams) as fin:
        brain_hparams = load_hyperpyyaml(fin, overrides)

    asr_brain = Seq2SeqASR(
        modules=brain_hparams["modules"],
        opt_class=brain_hparams["opt_class"],
        hparams=brain_hparams,
        run_opts=run_opts,
        checkpointer=brain_hparams["checkpointer"],
    )

    fmodel = fb.BrainModel(asr_brain, bounds=(-0.1,0.1))
    attack = fb.attacks.LinfPGD()
    epsilons = [0.0, 0.001]
    _, advs, success = attack(fmodel, batch, BrainMisclassification(predicted_words), epsilons=epsilons)

    adv_words, adv_tokens = asr_model.transcribe_batch(
            advs, rel_length
        )
    print(adv_words)