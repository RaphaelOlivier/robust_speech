"""
Evaluation script supporting adversarial attacks.
Similar to the training script without the brain.fit() call, with one key difference:
To support transferred attacks or attacks conducted on multiple models like MGAA,
the model hparams files was decoupled from the main hparams file.

hparams contains target_brain_class and target_brain_hparams_file arguments,
which are used to load corresponding brains, modules and pretrained parameters.
Optional source_brain_class and source_brain_hparams_file can be specified to transfer
the adversarial perturbations. Each of them can be specified as a (nested) list, in which case
the brain will be an EnsembleASRBrain object.

Example:
python evaluate.py attack_configs/pgd/attack.yaml\
     --root=/path/to/data/and/results/folder\
     --auto_mix_prec`
"""
import os
import sys
from pathlib import Path

import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main

import robust_speech as rs
from robust_speech.adversarial.brain import AdvASRBrain
from robust_speech.adversarial.utils import TargetGeneratorFromFixedTargets


def read_brains(
    brain_classes,
    brain_hparams,
    attacker=None,
    run_opts={},
    overrides={},
    tokenizer=None,
):
    if isinstance(brain_classes, list):
        brain_list = []
        assert len(brain_classes) == len(brain_hparams)
        for bc, bf in zip(brain_classes, brain_hparams):
            br = read_brains(
                bc, bf, run_opts=run_opts, overrides=overrides, tokenizer=tokenizer
            )
            brain_list.append(br)
        brain = rs.adversarial.brain.EnsembleASRBrain(brain_list)
    else:
        if isinstance(brain_hparams, str):
            with open(brain_hparams) as fin:
                brain_hparams = load_hyperpyyaml(fin, overrides)
        checkpointer = (
            brain_hparams["checkpointer"] if "checkpointer" in brain_hparams else None
        )
        brain = brain_classes(
            modules=brain_hparams["modules"],
            hparams=brain_hparams,
            run_opts=run_opts,
            checkpointer=checkpointer,
            attacker=attacker,
        )
        if "pretrainer" in brain_hparams:
            run_on_main(brain_hparams["pretrainer"].collect_files)
            brain_hparams["pretrainer"].load_collected(
                device=run_opts["device"])
        brain.tokenizer = tokenizer
    return brain


def evaluate(hparams_file, run_opts, overrides):
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    if "pretrainer" in hparams:  # load parameters
        # the tokenizer currently is loaded from the main hparams file and set
        # in all brain classes
        run_on_main(hparams["pretrainer"].collect_files)
        hparams["pretrainer"].load_collected(device=run_opts["device"])

    # Dataset prep (parsing Librispeech)
    prepare_dataset = hparams["dataset_prepare_fct"]

    # multi-gpu (ddp) save data preparation
    run_on_main(
        prepare_dataset,
        kwargs={
            "data_folder": hparams["data_folder"],
            "te_splits": hparams["test_splits"],
            "save_folder": hparams["csv_folder"],
            "skip_prep": hparams["skip_prep"],
        },
    )

    dataio_prepare = hparams["dataio_prepare_fct"]

    # here we create the datasets objects as well as tokenization and encoding
    _, _, test_datasets, _, _, tokenizer = dataio_prepare(hparams)
    source_brain = None
    if "source_brain_class" in hparams:  # loading source model
        source_brain = read_brains(
            hparams["source_brain_class"],
            hparams["source_brain_hparams_file"],
            run_opts=run_opts,
            overrides={"root": hparams["root"]},
            tokenizer=tokenizer,
        )
    attacker = hparams["attack_class"]
    if source_brain and attacker:
        # instanciating with the source model if there is one.
        # Otherwise, AdvASRBrain will handle instanciating the attacker with
        # the target model.
        attacker = attacker(source_brain)

    # Target model initialization
    target_brain_class = hparams["target_brain_class"]
    target_hparams = (
        hparams["target_brain_hparams_file"]
        if hparams["target_brain_hparams_file"]
        else hparams
    )
    target_brain = read_brains(
        target_brain_class,
        target_hparams,
        attacker=attacker,
        run_opts=run_opts,
        overrides={"root": hparams["root"]},
        tokenizer=tokenizer,
    )
    target_brain.logger = hparams["logger"]
    target_brain.hparams.train_logger = hparams["logger"]

    target = None
    if "target_generator" in hparams:
        target = hparams["target_generator"]
    elif "target_sentence" in hparams:
        target = TargetGeneratorFromFixedTargets(
            target=hparams["target_sentence"])
    load_audio = hparams["load_audio"] if "load_audio" in hparams else None
    save_audio_path = hparams["save_audio_path"] if hparams["save_audio"] else None
    # Evaluation
    for k in test_datasets.keys():  # keys are test_clean, test_other etc
        target_brain.hparams.wer_file = os.path.join(
            hparams["output_folder"], "wer_{}.txt".format(k)
        )
        target_brain.evaluate(
            test_datasets[k],
            test_loader_kwargs=hparams["test_dataloader_opts"],
            load_audio=load_audio,
            save_audio_path=save_audio_path,
            sample_rate=hparams["sample_rate"],
            target=target,
        )


if __name__ == "__main__":

    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    # If distributed_launch=True then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    evaluate(hparams_file, run_opts, overrides)
