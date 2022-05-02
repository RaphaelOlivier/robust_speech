#!/bin/bash

python universal_evaluate.py attack_configs/universal/ctc_5000bpe_universal.yaml --data_csv_name=test-clean-adv-100 --root=/home/deokhk/coursework/robust_speech/root --snr=30 --lr 0.001 --success_rate 0.35 --seed=1026
