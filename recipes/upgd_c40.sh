#!/bin/bash

python universal_evaluate.py attack_configs/pgd/ctc_5000bpe_universal.yaml --data_csv_name=test-clean-adv-100 --root=/home/deokhk/coursework/robust_speech/root --snr=40
