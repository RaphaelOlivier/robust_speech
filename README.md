# robust_speech

Adversarial attacks and defenses on Speech Recognition  - powered by [SpeechBrain](https://github.com/speechbrain/speechbrain).

Developed in the MLSP lab at CMU. Project led by [Raphael Olivier](https://raphaelolivier.github.io) under the supervision of Prof. Bhiksha Raj.

## Features 

* Run adversarial attacks over a dataset
* Evaluate with Word-Error Rate(WER), Character-Error Rate (CER) and Signal-to-Noise Ratio (SNR)
* Export adversarial examples
* Run an attack over multiple models at once
* Transfer an attack from a source model to a target model.
* Attacks on SpeechBrain models through the brain class over one file
* (In testing) Adversarial training

Supported attacks:
* [PGD](https://arxiv.org/abs/1706.06083)
* [CW](https://people.eecs.berkeley.edu/~daw/papers/audio-dls18.pdf)
* [Kenansville](https://arxiv.org/abs/1910.05262) 
* (In testing) [Imperceptible](https://arxiv.org/abs/1903.10346) 
* (In testing) [MGAA](https://arxiv.org/abs/2108.04204)
* (In testing) [Genetic](https://arxiv.org/abs/1801.00554)
* Wav2Vec2 [ACL](https://arxiv.org/abs/2006.07589)

The package provised model classes in the form of Speechbrain Brain classes, that are compatible with the attacks above. Currently implemented:
* Sequence-to-Sequence models with RNNs and Transformers
* CTC models (DeepSpeech-like) with RNNs and Transformers, subword and character tokenizers.
* RNN-Transducer models
* Wav2Vec2 ASR and Pretraining

We also provide data preparation and loading functions for the LibriSpeech package, based on SpeechBrain recipes.

Results of adversarial attacks on these models and this dataset will be shorly published in a preprint paper.

## Install 
TODO
## Use
TODO

## Incoming features
* Attacks:
    * [Universal](https://arxiv.org/abs/1905.03828)
    * [FAPG](https://www.aaai.org/AAAI21Papers/AAAI-7923.XieY.pdf)
* Randomized Smoothing for ASR (https://arxiv.org/abs/2112.03000)
* Data augmentation
* Data poisoning
* ASR attacks on video