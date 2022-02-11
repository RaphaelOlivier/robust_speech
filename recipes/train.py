
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
            "tr_splits": hparams["train_splits"],
            "dev_splits": hparams["dev_splits"],
            "te_splits": hparams["test_splits"],
            "save_folder": hparams["output_folder"],
            "merge_lst": hparams["train_splits"],
            "merge_name": "train.csv",
            "skip_prep": hparams["skip_prep"],
        },
    )

    dataio_prepare = hparams["dataio_prepare_fct"]

    # here we create the datasets objects as well as tokenization and encoding
    train_data, valid_data, test_datasets, tokenizer = dataio_prepare(
        hparams
    )
    # Trainer initialization
    brain_class = hparams["brain_class"]

    asr_brain = brain_class(
        modules=hparams["modules"],
        hparams=hparams,
        opt_class=hparams["opt_class"],
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
        attacker=hparams["attack_class"],
    )

    # We dynamicaly add the tokenizer to our brain class.
    # NB: This tokenizer corresponds to the one used for the LM!!
    asr_brain.tokenizer = tokenizer

    # Training
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )

    # Testing
    for k in test_datasets.keys():  # keys are test_clean, test_other etc
        asr_brain.hparams.wer_file = os.path.join(
            hparams["output_folder"], "wer_{}.txt".format(k)
        )
        asr_brain.evaluate(
            test_datasets[k], test_loader_kwargs=hparams["test_dataloader_opts"]
        )
