# robust_speech

Adversarial attacks and defenses on Speech Recognition  -powered by SpeechBrain.

## Features
* Attacks on SpeechBrain models through the brain class over one file
* PGD attack
* CW attack
* SNR, WER and CER evaluation
* Attacks verified on Seq2Seq ASR (CW has trouble converging)

## TODO
* Attack a dataset
* Try attacks on CTC, Transducer, Transformer and Wav2Vec2 model
* Transferability evaluation
* Attacks:
    * Wav2Vec2 ACL
    * [Kenansville](https://arxiv.org/abs/1910.05262) 
    * [Imperceptible](https://arxiv.org/abs/1903.10346)
    * [Universal](https://arxiv.org/abs/1905.03828)
    * [MGAA](https://arxiv.org/abs/2108.04204)
    
* Adversarial Training
* Randomized Smoothing
* Data augmentation
