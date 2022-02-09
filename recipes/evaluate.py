
import os
import sys
import gc
import torch
import logging
import speechbrain as sb
from robust_speech.adversarial.brain import AdvASRBrain
from speechbrain.utils.distributed import run_on_main
from hyperpyyaml import load_hyperpyyaml
from pathlib import Path
import robust_speech as rs

"""
Adversarial (or natural) training script
"""

def read_brains(brain_classes, brain_hparams,attacker=None):
    if isinstance(brain_classes,list):
        brain_list = []
        assert len(brain_classes) == len(brain_hparams)
        for bc,bf in zip(brain_classes,brain_hparams):
            br = read_brains(bc,bf)
            brain_list.append(br)
        brain = rs.adversarial.brain.EnsembleASRBrain(brain_list)
    else:
        if isinstance(brain_hparams,str):
            with open(brain_hparams) as fin:
                brain_hparams = load_hyperpyyaml(fin,{})
        brain = brain_classes(
            modules=brain_hparams["modules"],
            hparams=brain_hparams,
            run_opts=run_opts,
            checkpointer=None,
            attacker=attacker,
        )
    return brain

if __name__ == "__main__":

    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # If distributed_launch=True then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Dataset prep (parsing Librispeech)
    prepare_dataset = hparams["dataset_prepare_fct"]

    # multi-gpu (ddp) save data preparation
    run_on_main(
        prepare_dataset,
        kwargs={
            "data_folder": hparams["data_folder"],
            "te_splits": hparams["test_splits"],
            "save_folder": hparams["output_folder"],
            "skip_prep": hparams["skip_prep"],
        },
    )

    dataio_prepare = hparams["dataio_prepare_fct"]

    # here we create the datasets objects as well as tokenization and encoding
    _, _, test_datasets, tokenizer = dataio_prepare(
        hparams
    )

    source_brain = None
    if hparams["source_brain_class"]:
        source_brain = read_brains(hparams["source_brain_class"], hparams["source_brain_hparams_file"])
    attacker=hparams["attack_class"]
    if source_brain:
        attacker = attacker(source_brain)

    # Target model initialization
    target_brain_class = hparams["target_brain_class"]
    target_hparams = hparams["target_brain_hparams_file"] if hparams["target_brain_hparams_file"] else hparams
    target_brain = read_brains(target_brain_class, target_hparams, attacker=attacker)
    target_brain.__setattr__("tokenizer",tokenizer, attacker_brain=True)
    # Testing
    for k in test_datasets.keys():  # keys are test_clean, test_other etc
        target_brain.hparams.wer_file = os.path.join(
            hparams["output_folder"], "wer_{}.txt".format(k)
        )
        target_brain.evaluate(
            test_datasets[k], test_loader_kwargs=hparams["test_dataloader_opts"]
        )
