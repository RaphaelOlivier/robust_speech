#!/bin/bash

python universal_evaluate.py attack_configs/pgd/ctc_5000bpe_universal.yaml --data_csv_name=test-clean-short --root=/home/deokhk/coursework/robust_speech/root --snr=30 --seed=1026
