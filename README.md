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
* Sequence-to-Sequence models with RNNs and Transformers, also supporting the CTC loss. Compatible with [pretrained](https://huggingface.co/speechbrain/asr-crdnn-rnnlm-librispeech) [speechbrain](https://huggingface.co/speechbrain/asr-crdnn-transformerlm-librispeech) [models](https://huggingface.co/speechbrain/asr-transformer-transformerlm-librispeech).
* RNN-CTC model (DeepSpeech-like) with character tokenizer.
* RNN-Transducer models
* Wav2Vec2 ASR and Pretraining (compatible with pretrained huggingface Wav2Vec2 models).

We also provide data preparation and loading functions for the LibriSpeech package, based on SpeechBrain recipes.

## Install 
Before installing robust_speech you should have installed PyTorch (>=1.8.0,<=1.10.1) with cuda support in your environment.

```
git clone https://github.com/RaphaelOlivier/robust_speech.git
cd robust_speech
pip install .
```

## Usage
We provide scripts to train and evaluate models (with adversarial attacks in both stages). These scripts are similar to speechbrain training recipes.

We also provide a number of training and evaluation config files ready for use.

Example

```
# in ./recipes/

# This will download the speechbrain/asr-crdnn-rnnlm-librispeech model from huggingface
python evaluate.py attack_configs/pgd/s2s_1000bpe.yaml --root=/path/to/results/folder 

# This will train a model first
python train.py train_configs/ctc_train.yaml --root=/path/to/results/folder
mv /path/to/training/outputs/folder/*.ckpt /path/to/models/folder/asr-ctcrnn-librispeech/
python evaluate.py attack_configs/pgd/ctc_char.yaml --root=/path/to/results/folder --snr=25
```

Our provided configs assume a results folder structure such as
```
root
│
└───data
│   │ # where datasets are dumped (e.g. download LibriSpeech here)
│
└───models
│   │
│   └───model_name1
│   │   │   model.ckpt
│   
└───tokenizers   
│   │ # where all tokenizers are saved
│   
└───trainings
│   │  # where your custom models are trained
│  
└───attacks
|   |
│   └───attack_name
│   │   │
│   │   └───1234 # seed
│   │   │   │
│   │   │   └───model_name1
│   │   │   │   │ # your attack results

```
You may change this at will in your custom `.yaml` files or with command line arguments
## Incoming features
* Attacks:
    * [Universal](https://arxiv.org/abs/1905.03828)
    * [FAPG](https://www.aaai.org/AAAI21Papers/AAAI-7923.XieY.pdf)
* Randomized [Smoothing for ASR](https://arxiv.org/abs/2112.03000)
* Data augmentation
* Data poisoning

## Cite
Results of adversarial attacks on these models and this dataset will be shorly published in a preprint paper. Incoming...
