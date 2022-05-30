"""
Data preparation and loading scripts for CommonVoice.
The prepare_common_voice function was copied from 
https://github.com/speechbrain/speechbrain/blob/develop/recipes/CommonVoice/common_voice_prepare.py
"""

import os
import csv
import re
import logging
import torchaudio
import unicodedata
from tqdm.contrib import tzip

logger = logging.getLogger(__name__)


def prepare_common_voice(
    data_folder,
    save_folder,
    tr_splits=[],
    dev_splits=[],
    te_splits=[],
    accented_letters=False,
    language="en",
    skip_prep=False,
    **kwargs,
):
    """
    Prepares the csv files for the Mozilla Common Voice dataset.
    Download: https://voice.mozilla.org/en/datasets
    Arguments
    ---------
    data_folder : str
        Path to the folder where the original Common Voice dataset is stored.
        This path should include the lang: /datasets/CommonVoice/en/
    save_folder : str
        The directory where to store the csv files.
    tr_splits : list, optional
        Path to the Train Common Voice .tsv file(s) (cs)
    dev_splits : list, optional
        Path to the Dev Common Voice .tsv file(s) (cs)
    te_splits : list, optional
        Path to the Test Common Voice .tsv file(s) (cs)
    accented_letters : bool, optional
        Defines if accented letters will be kept as individual letters or
        transformed to the closest non-accented letters.
    language: str
        Specify the language for text normalization.
    skip_prep: bool
        If True, skip data preparation.
    Example
    -------
    >>> from recipes.CommonVoice.common_voice_prepare import prepare_common_voice
    >>> data_folder = '/datasets/CommonVoice/en'
    >>> save_folder = 'exp/CommonVoice_exp'
    >>> train_tsv_file = '/datasets/CommonVoice/en/train.tsv'
    >>> dev_tsv_file = '/datasets/CommonVoice/en/dev.tsv'
    >>> test_tsv_file = '/datasets/CommonVoice/en/test.tsv'
    >>> accented_letters = False
    >>> duration_threshold = 10
    >>> prepare_common_voice( \
                 data_folder, \
                 save_folder, \
                 train_tsv_file, \
                 dev_tsv_file, \
                 test_tsv_file, \
                 accented_letters, \
                 language="en" \
                 )
    """

    splits = tr_splits + dev_splits + te_splits

    # Setting the save folder
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    if skip(splits, save_folder, {}) and skip_prep:
        logger.info("Skipping preparation, completed in previous run.")
        return
    else:
        logger.info("Data_preparation...")

    tsv_splits = [os.path.join(data_folder, s+'.tsv') for s in splits]

    save_csv_splits = [os.path.join(save_folder, s+'.csv') for s in splits]

    # Additional checks to make sure the data folder contains Common Voice
    check_commonvoice_folders(data_folder)

    for i in range(len(splits)):
        assert os.path.exists(
            tsv_splits[i]), "tsv file must be available for data preparation"
        create_csv(
            tsv_splits[i],
            save_csv_splits[i],
            data_folder,
            accented_letters,
            language,
        )


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


def create_csv(
    orig_tsv_file, csv_file, data_folder, accented_letters=False, language="en"
):
    """
    Creates the csv file given a list of wav files.
    Arguments
    ---------
    orig_tsv_file : str
        Path to the Common Voice tsv file (standard file).
    data_folder : str
        Path of the CommonVoice dataset.
    accented_letters : bool, optional
        Defines if accented letters will be kept as individual letters or
        transformed to the closest non-accented letters.
    Returns
    -------
    None
    """

    # Check if the given files exists
    if not os.path.isfile(orig_tsv_file):
        msg = "\t%s doesn't exist, verify your dataset!" % (orig_tsv_file)
        logger.info(msg)
        raise FileNotFoundError(msg)

    # We load and skip the header
    loaded_csv = open(orig_tsv_file, "r").readlines()[1:]
    nb_samples = str(len(loaded_csv))

    msg = "Preparing CSV files for %s samples ..." % (str(nb_samples))
    logger.info(msg)

    # Adding some Prints
    msg = "Creating csv lists in %s ..." % (csv_file)
    logger.info(msg)

    csv_lines = [["ID", "duration", "wav", "spk_id", "wrd"]]
    # Start processing lines
    total_duration = 0.0
    for line in tzip(loaded_csv):
        line = line[0]
        # Path is at indice 1 in Common Voice tsv files. And .mp3 files
        # are located in datasets/lang/clips/
        mp3_path = os.path.join(data_folder, "clips", line.split("\t")[1])
        file_name = mp3_path.split(".")[-2].split("/")[-1]
        spk_id = line.split("\t")[0]
        snt_id = file_name

        # Setting torchaudio backend to sox-io (needed to read mp3 files)
        if torchaudio.get_audio_backend() != "sox_io":
            logger.warning(
                "This recipe needs the sox-io backend of torchaudio")
            logger.warning("The torchaudio backend is changed to sox_io")
            torchaudio.set_audio_backend("sox_io")

        # Reading the signal (to retrieve duration in seconds)
        if os.path.isfile(mp3_path):
            info = torchaudio.info(mp3_path)
            audio_path = mp3_path
        else:
            wav_path = os.path.join(
                data_folder, "clips", "wav", line.split("\t")[1][:-4]+'.wav')
            file_name = wav_path.split(".")[-2].split("/")[-1]
            if os.path.isfile(wav_path):
                info = torchaudio.info(wav_path)
                audio_path = wav_path
            else:
                msg = "\tError loading: %s" % (file_name)
                logger.info(msg)
                continue

        duration = info.num_frames / info.sample_rate
        total_duration += duration

        # Getting transcript
        words = line.split("\t")[2]

        # Unicode Normalization
        words = unicode_normalisation(words)

        # !! Language specific cleaning !!
        # Important: feel free to specify the text normalization
        # corresponding to your alphabet.

        if language in ["en", "fr", "it", "rw"]:
            words = re.sub(
                "[^’'A-Za-z0-9À-ÖØ-öø-ÿЀ-ӿéæœâçèàûî]+", " ", words
            ).upper()

        if language == "fr":
            # Replace J'y D'hui etc by J_ D_hui
            words = words.replace("'", " ")
            words = words.replace("’", " ")

        elif language == "ar":
            HAMZA = "\u0621"
            ALEF_MADDA = "\u0622"
            ALEF_HAMZA_ABOVE = "\u0623"
            letters = (
                "ابتةثجحخدذرزسشصضطظعغفقكلمنهويىءآأؤإئ"
                + HAMZA
                + ALEF_MADDA
                + ALEF_HAMZA_ABOVE
            )
            words = re.sub("[^" + letters + " ]+", "", words).upper()
        elif language == "ga-IE":
            # Irish lower() is complicated, but upper() is nondeterministic, so use lowercase
            def pfxuc(a):
                return len(a) >= 2 and a[0] in "tn" and a[1] in "AEIOUÁÉÍÓÚ"

            def galc(w):
                return w.lower() if not pfxuc(w) else w[0] + "-" + w[1:].lower()

            words = re.sub("[^-A-Za-z'ÁÉÍÓÚáéíóú]+", " ", words)
            words = " ".join(map(galc, words.split(" ")))

        # Remove accents if specified
        if not accented_letters:
            words = strip_accents(words)
            words = words.replace("'", " ")
            words = words.replace("’", " ")

        # Remove multiple spaces
        words = re.sub(" +", " ", words)

        # Remove spaces at the beginning and the end of the sentence
        words = words.lstrip().rstrip()

        # Getting chars
        chars = words.replace(" ", "_")
        chars = " ".join([char for char in chars][:])

        # Remove too short sentences (or empty):
        if len(words.split(" ")) < 3:
            continue

        # Composition of the csv_line
        csv_line = [snt_id, str(duration), audio_path, spk_id, str(words)]

        # Adding this line to the csv_lines list
        csv_lines.append(csv_line)
    # Writing the csv lines
    with open(csv_file, mode="w", encoding="utf-8") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )

        for line in csv_lines:
            csv_writer.writerow(line)
    # Final prints
    msg = "%s successfully created!" % (csv_file)
    logger.info(msg)
    msg = "Number of samples: %s " % (str(len(loaded_csv)))
    logger.info(msg)
    msg = "Total duration: %s Hours" % (str(round(total_duration / 3600, 2)))
    logger.info(msg)


def check_commonvoice_folders(data_folder):
    """
    Check if the data folder actually contains the Common Voice dataset.
    If not, raises an error.
    Returns
    -------
    None
    Raises
    ------
    FileNotFoundError
        If data folder doesn't contain Common Voice dataset.
    """

    files_str = "/clips"

    # Checking clips
    if not os.path.exists(data_folder + files_str):

        err_msg = (
            "the folder %s does not exist (it is expected in "
            "the Common Voice dataset)" % (data_folder + files_str)
        )
        raise FileNotFoundError(err_msg)


def unicode_normalisation(text):

    try:
        text = unicode(text, "utf-8")
    except NameError:  # unicode is a default on python 3
        pass
    return str(text)


def strip_accents(text):

    text = (
        unicodedata.normalize("NFD", text)
        .encode("ascii", "ignore")
        .decode("utf-8")
    )

    return str(text)
