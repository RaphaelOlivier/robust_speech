"""
Dataio preparation and loading scripts for all ASR models.
The dataio_prepare is a mixture from that function in 
speechbrain recipes using subwords tokenizers
(https://github.com/speechbrain/speechbrain/tree/develop/recipes/LibriSpeech/ASR/seq2seq/train.py)
and Char tokenizers 
(https://github.com/speechbrain/speechbrain/blob/develop/recipes/LibriSpeech/ASR/CTC/train_with_wav2vec.py)
in order to handle both cases.
"""

import csv
import logging
import os
import random
from collections import Counter
from pathlib import Path

import sentencepiece
import speechbrain as sb
import torch
import torchaudio
from speechbrain.dataio.dataio import load_pkl, merge_csvs, save_pkl
from speechbrain.utils.data_utils import download_file, get_all_files

logger = logging.getLogger(__name__)
OPT_FILE = "opt_librispeech_prepare.pkl"
SAMPLERATE = 16000


def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""
    data_folder = hparams["data_folder"]

    train_data = None
    if "train_csv" in hparams:
        train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=hparams["train_csv"],
            replacements={"data_root": data_folder},
        )

        train_data = train_data.filtered_sorted(
            key_max_value={"duration": hparams["avoid_if_longer_than"]}
        )

        if hparams["sorting"] == "ascending":
            # we sort training data to speed up training and get better
            # results.
            train_data = train_data.filtered_sorted(sort_key="duration")
            # when sorting do not shuffle in dataloader ! otherwise is
            # pointless
            hparams["train_dataloader_opts"]["shuffle"] = False

        elif hparams["sorting"] == "descending":
            train_data = train_data.filtered_sorted(
                sort_key="duration", reverse=True)
            # when sorting do not shuffle in dataloader ! otherwise is
            # pointless
            hparams["train_dataloader_opts"]["shuffle"] = False

        elif hparams["sorting"] == "random":
            pass

        else:
            raise NotImplementedError(
                "sorting must be random, ascending or descending")

    valid_data = None
    if "valid_csv" in hparams:
        valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=hparams["valid_csv"],
            replacements={"data_root": data_folder},
        )
        # valid_data = valid_data.filtered_sorted(sort_key="duration")

        valid_data = valid_data.filtered_sorted(
            key_max_value={"duration": hparams["avoid_if_longer_than"]}
        )

    # test is separate
    test_datasets = {}
    if "test_csv" in hparams:
        for csv_file in hparams["test_csv"]:
            name = Path(csv_file).stem
            test_datasets[name] = sb.dataio.dataset.DynamicItemDataset.from_csv(
                csv_path=csv_file, replacements={"data_root": data_folder}
            )
            test_datasets[name] = test_datasets[name].filtered_sorted(
                sort_key="duration", reverse=True
            )
            if "avoid_if_longer_than" in hparams:
                test_datasets[name] = test_datasets[name].filtered_sorted(
                    key_max_value={"duration": hparams["avoid_if_longer_than"]}
                )
    datasets = []
    if train_data:
        datasets.append(train_data)
    if valid_data:
        datasets.append(valid_data)
    datasets = datasets + [i for k, i in test_datasets.items()]

    # We get the tokenizer as we need it to encode the labels when creating
    # mini-batches.
    tokenizer = hparams["tokenizer"]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):

        info = torchaudio.info(wav)
        sig = sb.dataio.dataio.read_audio(wav)
        if info.num_channels > 1:
            sig = torch.mean(sig, dim=1)
        if info.sample_rate != hparams["sample_rate"]:
            sig = torchaudio.transforms.Resample(
                info.sample_rate, hparams["sample_rate"],
            )(sig)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    if isinstance(tokenizer, sb.dataio.encoder.CTCTextEncoder):  # char encoder
        tokenizer.add_unk()

        @sb.utils.data_pipeline.takes("wrd")
        @sb.utils.data_pipeline.provides(
            "wrd", "char_list", "tokens_list", "tokens_bos", "tokens_eos", "tokens"
        )
        def text_pipeline(wrd):
            yield wrd
            char_list = list(wrd)
            yield char_list
            tokens_list = tokenizer.encode_sequence(char_list)
            yield tokens_list
            tokens_bos = torch.LongTensor(
                [hparams["bos_index"]] + (tokens_list))
            yield tokens_bos
            tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
            yield tokens_eos
            tokens = torch.LongTensor(tokens_list)
            yield tokens

        sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

        if "pretrainer" in hparams and "tokenizer" in hparams["pretrainer"].loadables:
            # tokenizer has already been loaded
            pass
        else:
            lab_enc_file = os.path.join(
                hparams["save_folder"], "label_encoder.txt")

            special_labels = {
                "bos_label": hparams["bos_index"],
                "eos_label": hparams["eos_index"],
                "blank_label": hparams["blank_index"],
            }
            tokenizer.load_or_create(
                path=lab_enc_file,
                from_didatasets=[datasets[0]],
                output_key="char_list",
                special_labels=special_labels,
                sequence_input=True,
            )

        # 4. Set output:
        sb.dataio.dataset.set_output_keys(
            datasets,
            ["id", "sig", "wrd", "char_list", "tokens_bos", "tokens_eos", "tokens"],
        )
    else:
        assert isinstance(tokenizer, sentencepiece.SentencePieceProcessor)

        @sb.utils.data_pipeline.takes("wrd")
        @sb.utils.data_pipeline.provides(
            "wrd", "tokens_list", "tokens_bos", "tokens_eos", "tokens"
        )
        def text_pipeline(wrd):
            yield wrd
            tokens_list = tokenizer.encode_as_ids(wrd)
            yield tokens_list
            tokens_bos = torch.LongTensor(
                [hparams["bos_index"]] + (tokens_list))
            yield tokens_bos
            tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
            yield tokens_eos
            tokens = torch.LongTensor(tokens_list)
            yield tokens

        sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

        # 4. Set output:
        sb.dataio.dataset.set_output_keys(
            datasets,
            ["id", "sig", "wrd", "tokens_bos", "tokens_eos", "tokens"],
        )

    train_batch_sampler = None
    valid_batch_sampler = None
    if "dynamic_batching" in hparams and hparams["dynamic_batching"]:
        from speechbrain.dataio.batch import PaddedBatch  # noqa
        from speechbrain.dataio.dataloader import SaveableDataLoader  # noqa
        from speechbrain.dataio.sampler import DynamicBatchSampler  # noqa

        dynamic_hparams = hparams["dynamic_batch_sampler"]

        num_buckets = dynamic_hparams["num_buckets"]

        max_batch_len = dynamic_hparams["max_batch_len"]

        if train_data:
            train_batch_sampler = DynamicBatchSampler(
                train_data,
                max_batch_len,
                num_buckets=num_buckets,
                length_func=lambda x: x["duration"] * hparams["sample_rate"],
                shuffle=dynamic_hparams["shuffle_ex"],
                batch_ordering=dynamic_hparams["batch_ordering"],
            )

    return (
        train_data,
        valid_data,
        test_datasets,
        train_batch_sampler,
        valid_batch_sampler,
        tokenizer,
    )
