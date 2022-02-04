# robust_speech

Adversarial attacks and defenses on Speech Recognition  -powered by SpeechBrain.

## Features
* Attacks on SpeechBrain models through the brain class over one file
* [PGD](https://arxiv.org/abs/1706.06083) attack
* [CW](https://people.eecs.berkeley.edu/~daw/papers/audio-dls18.pdf) attack
* [Kenansville](https://arxiv.org/abs/1910.05262) 
* [Imperceptible](https://arxiv.org/abs/1903.10346)
* SNR, WER and CER evaluation
* Attacks verified on Seq2Seq ASR (CW has trouble converging)
* Attacks verified on CTC ASR (CW works, Imperceptible has trouble converging)
* Adversarial Training brain class
* Attack a dataset (evaluate AdvASRBrain)

## TODO
* Try attacks on Transducer, Transformer and Wav2Vec2 model
* Transferability evaluation
* Attacks:
    * Wav2Vec2 [ACL](https://arxiv.org/abs/2006.07589)
    * [Universal](https://arxiv.org/abs/1905.03828)
    * [MGAA](https://arxiv.org/abs/2108.04204)
    * [FAPG](https://www.aaai.org/AAAI21Papers/AAAI-7923.XieY.pdf)
    * [Genetic](https://arxiv.org/abs/1801.00554)
    
* Randomized Smoothing
* Data augmentation
* Data poisoning
* ASR attacks on video
* ROVER in python (?)