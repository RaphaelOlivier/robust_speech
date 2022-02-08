
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
        if isinstance(hparams["source_brain_class"],list):
            source_brain_list = []
            assert len(hparams["source_brain_class"]) == len(hparams["source_brain_hparams_file"])
            for sbc,sbf in zip(hparams["source_brain_class"],hparams["source_brain_hparams_file"]):
                with open(sbf) as fin:
                    sbh = load_hyperpyyaml(fin,{})
                sbo = sbc(
                    modules=sbh["modules"],
                    hparams=sbh,
                    run_opts=run_opts,
                    checkpointer=None,
                    attacker=None,
                )
                source_brain_list.append(sbo)
            source_brain = rs.adversarial.brain.EnsembleASRBrain(source_brain_list)
        else:
            with open(hparams["source_brain_hparams_file"]) as fin:
                source_brain_hparams = load_hyperpyyaml(fin,{})
            source_brain = hparams["source_brain_class"](
                modules=source_brain_hparams["modules"],
                hparams=source_brain_hparams,
                run_opts=run_opts,
                checkpointer=None,
                attacker=None,
            )
    attacker=hparams["attack_class"]
    if source_brain:
        attacker = attacker(source_brain)

    # Target model initialization
    target_brain_class = hparams["target_brain_class"]
    if hparams["target_brain_hparams_file"] is not None:
        with open(hparams["target_brain_hparams_file"]) as fin:
            target_brain_hparams = load_hyperpyyaml(fin, {})
    else:
        target_brain_hparams = hparams

    target_brain = target_brain_class(
        modules=target_brain_hparams["modules"],
        hparams=target_brain_hparams,
        run_opts=run_opts,
        checkpointer=None,
        attacker=hparams["attack_class"],
    )
    target_brain.tokenizer = target_brain_hparams["tokenizer"]

    # Testing
    for k in test_datasets.keys():  # keys are test_clean, test_other etc
        target_brain.hparams.wer_file = os.path.join(
            hparams["output_folder"], "wer_{}.txt".format(k)
        )
        target_brain.evaluate(
            test_datasets[k], test_loader_kwargs=hparams["test_dataloader_opts"]
        )
