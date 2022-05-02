#!/bin/bash

python universal_evaluate.py attack_configs/universal/ctc_5000bpe_universal.yaml --data_csv_name=test-clean-short --root=/home/deokhk/coursework/robust_speech/root --nb_iter=10 --eps=0.2 --lr 0.001 --seed=1026
