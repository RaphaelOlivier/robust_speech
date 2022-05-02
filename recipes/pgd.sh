#!/bin/bash
python evaluate.py attack_configs/pgd/ctc_5000bpe.yaml --data_csv_name=test-clean-short-train --root=/home/deokhk/coursework/robust_speech/root --snr=30 --seed=1026
