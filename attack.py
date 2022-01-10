
import sys
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.pretrained import EncoderDecoderASR


if __name__ == "__main__":
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        attack_hparams = load_hyperpyyaml(fin, overrides)

    models_path = "/home/raphael/dataspace/models/robust_speech/"
    asr_model = EncoderDecoderASR.from_hparams(
        source=attack_hparams["source"], 
        hparams_file= attack_hparams["hparams_file"], 
        savedir=attack_hparams["savedir"]
    )
    audio_file = attack_hparams["input"]
    string=asr_model.transcribe_file(audio_file)
    print(string)