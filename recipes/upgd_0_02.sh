#!/bin/bash

python universal_evaluate.py attack_configs/pgd_inf/ctc_5000bpe_universal.yaml --data_csv_name=test-clean-adv-100 --root=/home/deokhk/coursework/robust_speech/root --eps=0.02
