#!/bin/bash

python /home/deokhk/coursework/robust_speech/recipes/universal_evaluate.py attack_configs/universal/ctc_5000bpe_universal.yaml --data_csv_name=test-clean-adv-100 --root=/home/deokhk/coursework/robust_speech/root --nb_iter=30 --eps=0.025 --lr 0.001 --seed=1026 --time_universal=True