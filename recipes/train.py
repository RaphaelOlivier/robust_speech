
"""
Training script for robust-speech models. 
It handles adversarial training using the hparams.attacker object, 
and externalizes data loading and brain classes with hparams.
This aside, the script is very similar to SpeechBrain training scripts. 
It is compatible with any robust-speech model.

Example:

`python train.py train_configs/ctc_train.yaml\
     --root=/path/to/data/and/results/folder\
     --auto_mix_prec\
     --data_parallel_backend`
"""


import robust_speech as rs
from pathlib import Path
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main
from robust_speech.adversarial.brain import AdvASRBrain
import speechbrain as sb
import os
import sys
import logging
logger = logging.getLogger('speechbrain.dataio.sampler')
logger.setLevel(logging.WARNING)  # avoid annoying logs

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
    # data preparation function. Have skip_prep=True if csv files have already been processed.
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

    dataio_prepare = hparams["dataio_prepare_fct"]  # data loading function

    # load parameters (such as tokenizer or language model)
    if "pretrainer" in hparams:
        run_on_main(hparams["pretrainer"].collect_files)
        hparams["pretrainer"].load_collected(device=run_opts["device"])

    # here we create the datasets objects as well as tokenization and encoding
    train_data, valid_data, test_datasets, train_bsampler, valid_bsampler, tokenizer = dataio_prepare(
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

    train_dataloader_opts = hparams["train_dataloader_opts"]
    valid_dataloader_opts = hparams["valid_dataloader_opts"]

    if train_bsampler is not None:
        train_dataloader_opts = {"batch_sampler": train_bsampler}
    if valid_bsampler is not None:
        valid_dataloader_opts = {"batch_sampler": valid_bsampler}

    # Training (with attacks if hparams.attacker is not None)
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=train_dataloader_opts,
        valid_loader_kwargs=valid_dataloader_opts,
    )

    # Testing (with attacks if hparams.attacker is not None)
    for k in test_datasets.keys():  # keys are test_clean, test_other etc
        asr_brain.hparams.wer_file = os.path.join(
            hparams["output_folder"], "wer_{}.txt".format(k)
        )
        asr_brain.evaluate(
            test_datasets[k], test_loader_kwargs=hparams["test_dataloader_opts"]
        )
