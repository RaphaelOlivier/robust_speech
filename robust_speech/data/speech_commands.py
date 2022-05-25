"""
Data preparation and loading scripts for SpeechCommands.
"""

import os
import csv
import random
import re
import logging
import torchaudio
import unicodedata
from tqdm.contrib import tzip

logger = logging.getLogger(__name__)


def prepare_speech_commands(
    data_folder,
    save_folder,
    tr_splits=[],
    dev_splits=[],
    te_splits=[],
    skip_prep=False,
):

    splits = tr_splits + dev_splits + te_splits

    # Setting the save folder
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    if skip(splits, save_folder, {}) and skip_prep:
        logger.info("Skipping preparation, completed in previous run.")
        return
    else:
        logger.info("Data_preparation...")

    dataset = torchaudio.datasets.SPEECHCOMMANDS(data_folder, download=True)
    all_files = load_files_list(dataset._path)
    files_splits = split_files(dataset._path, all_files, splits)
    for i in range(len(splits)):
        csv_file = os.path.join(save_folder, splits[i]+'.csv')
        create_csv(files_splits[i], csv_file, dataset._path)


def skip(splits, save_folder, conf):
    """
    Detect when the librispeech data prep can be skipped.

    Arguments
    ---------
    splits : list
        A list of the splits expected in the preparation.
    save_folder : str
        The location of the seave directory
    conf : dict
        The configuration options to ensure they haven't changed.

    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """

    # Checking csv files
    skip = True

    for split in splits:
        if not os.path.isfile(os.path.join(save_folder, split + ".csv")):
            skip = False
    return skip


def load_files_list(root):
    classes = os.listdir(root)
    classes = [c for c in classes if os.path.isdir(os.path.join(root, c))]
    all_files = []
    for c in classes:
        folder = os.path.join(root, c)
        wavs_list = os.listdir(folder)
        for w in wavs_list:
            if w.endswith('.wav'):
                path = os.path.join(c, w)
                ID = c+'_'+w.split('.')[0]
                all_files.append((ID, path, c))
    return all_files


def split_files(root, allfiles, split_names):
    validation_txt = os.path.join(root, 'validation_list.txt')
    testing_txt = os.path.join(root, 'testing_list.txt')
    validation_list = []
    testing_list = []
    with open(validation_txt, 'r') as f:
        for line in f:
            validation_list.append(line[:-1])
    with open(testing_txt, 'r') as f:
        for line in f:
            testing_list.append(line[:-1])

    validation_set = set(validation_list)
    testing_set = set(testing_list)
    training, validation, testing = [], [], []
    for t in allfiles:
        if t[1] in validation_set:
            validation.append(t)
        elif t[1] in testing_set:
            testing.append(t)
        else:
            training.append(t)

    d = {"training": training, "validation": validation, "testing": testing}
    splits = [d[s] for s in split_names]
    return splits


def create_csv(
    file_list, csv_file, data_folder


):
    """
    Creates the csv file given a list of wav files.
    Arguments
    ---------
    orig_txt_file : str
        Path to the Common Voice txt file (standard file).
    data_folder : str
        Path of the dataset.
    Returns
    -------
    None
    """

    assert len(file_list) > 0, "no files to add!"
    random.shuffle(file_list)
    msg = "Preparing CSV files for %s samples ..." % (str(len(file_list)))
    logger.info(msg)

    # Adding some Prints
    msg = "Creating csv lists in %s ..." % (csv_file)
    logger.info(msg)

    csv_lines = [["ID", "duration", "wav", "wrd"]]

    # Start processing lines
    total_duration = 0.0
    for t in file_list:

        snt_id, wav, transcription = t
        file_name = os.path.basename(wav)
        filepath = os.path.join(data_folder, wav)

        # Setting torchaudio backend to sox-io (needed to read mp3 files)
        if torchaudio.get_audio_backend() != "sox_io":
            logger.warning(
                "This recipe needs the sox-io backend of torchaudio")
            logger.warning("The torchaudio backend is changed to sox_io")
            torchaudio.set_audio_backend("sox_io")

        # Reading the signal (to retrieve duration in seconds)
        if os.path.isfile(filepath):
            info = torchaudio.info(filepath)
        else:
            msg = "\tError loading: %s" % (str(len(file_name)))
            logger.info(msg)
            continue

        duration = info.num_frames / info.sample_rate
        total_duration += duration

        # Getting transcript

        if transcription.startswith('_'):
            continue  # ignore background noise

        words = transcription

        # Unicode Normalization
        words = unicode_normalisation(words)

        words = re.sub(
            "[^’'A-Za-z0-9À-ÖØ-öø-ÿЀ-ӿéæœâçèàûî]+", " ", words
        ).upper()

        # Remove multiple spaces
        words = re.sub(" +", " ", words)

        # Remove spaces at the beginning and the end of the sentence
        words = words.lstrip().rstrip()

        # Getting chars
        chars = words.replace(" ", "_")
        chars = " ".join([char for char in chars][:])
        assert len(words.split(" ")) == 1
        # Composition of the csv_line
        csv_line = [snt_id, str(duration), filepath, str(words)]

        # Adding this line to the csv_lines list
        csv_lines.append(csv_line)

    # Writing the csv lines
    print(csv_file)
    with open(csv_file, mode="w", encoding="utf-8") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )

        for line in csv_lines:
            csv_writer.writerow(line)

    # Final prints
    msg = "%s successfully created!" % (csv_file)
    logger.info(msg)
    msg = "Total duration: %s Hours" % (str(round(total_duration / 3600, 2)))
    logger.info(msg)


def unicode_normalisation(text):

    try:
        text = unicode(text, "utf-8")
    except NameError:  # unicode is a default on python 3
        pass
    return str(text)
