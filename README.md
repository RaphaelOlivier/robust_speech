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
    * Wav2Vec2 [ACL](https://arxiv.org/abs/2006.07589)
    * [Kenansville](https://arxiv.org/abs/1910.05262) 
    * [Imperceptible](https://arxiv.org/abs/1903.10346)
    * [Universal](https://arxiv.org/abs/1905.03828)
    * [MGAA](https://arxiv.org/abs/2108.04204)
    * [FAPG](https://www.aaai.org/AAAI21Papers/AAAI-7923.XieY.pdf)
    
* Adversarial Training
* Randomized Smoothing
* Data augmentation
