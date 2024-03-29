# robust_speech

Adversarial attacks and defenses on Speech Recognition  - powered by [SpeechBrain](https://github.com/speechbrain/speechbrain).

Developed in the MLSP lab at CMU. Project led by [Raphael Olivier](https://raphaelolivier.github.io) under the supervision of Prof. Bhiksha Raj.

## What is this package for?
In order to deploy machine learning models in the real world, it is important to make them robust to adversarial perturbations. This includes Automatic Speech Recognition models, which are [well known](https://arxiv.org/abs/2007.06622) to be vulnerable to adversarial examples. Many adversarial attacks and defenses have been developped by various research teams. In order to let users evaluate the robustness of their models, and come up more easily with new defenses, gathering all these attacks under a common codebase can be handy.

Several robustness-oriented packages already exist ([Advertorch](https://github.com/BorealisAI/advertorch), [ART](https://github.com/Trusted-AI/adversarial-robustness-toolbox/tree/main/art), [Foolbox](https://github.com/bethgelab/foolbox), [Cleverhans](https://github.com/cleverhans-lab/cleverhans), etc). However, nearly all of these packages suppose that the model performs image classification. Yet applying standard classification-based attacks is not trivial, as ASR models are typically more complex than a stack of PyTorch layers: they must handle variable length inputs, are trained with tricky losses, contain recurrent networks, etc. Some attacks have been developped with the explicit goal of fooling ASR models (for instance by relying on acoustic models), and these attacks are rarely included in general robustness packages. Out of the packages above, ART supports ASR attacks, and only a small subset of them and limited ASR architectures and datasets.

**robust_speech** fills that gap and propose a simple way to evaluate ASR models against Adversarial attacks. It is based on [Speechbrain](https://speechbrain.github.io/), a flexible Speech Toolkit that makes it very easy to load ASR datasets, run ASR models and access their loss, predictions and error rates - all of which are often necessary to run attacks. We have added some useful features and metrics to the Speechbrain Brain class, and we have imported or reproduced multiple general and ASR-specific attacks.

## Features 

* Run adversarial attacks over a dataset
* Evaluate with Word-Error Rate(WER), Character-Error Rate (CER) and Signal-to-Noise Ratio (SNR)
* Export adversarial examples
* Run an attack over multiple models at once
* Train and run attacks with trainable parameters
* Transfer an attack from a source model to a target model.
* Attacks on SpeechBrain models through the brain class over one file
* Defenses:
    * Randomized [Smoothing for ASR](https://arxiv.org/abs/2112.03000)
    * (In development) Adversarial training

Supported attacks:
* [PGD](https://arxiv.org/abs/1706.06083)
* [CW](https://people.eecs.berkeley.edu/~daw/papers/audio-dls18.pdf)
* [Kenansville](https://arxiv.org/abs/1910.05262)
* [Yeehaw Junction](https://arxiv.org/abs/2203.05408) (In development)
* [Imperceptible](https://arxiv.org/abs/1903.10346) 
* [Genetic](https://arxiv.org/abs/1801.00554)
* [Self-supervised attacks](https://arxiv.org/abs/2111.04330) for Wav2Vec2-type models
* [MGAA](https://arxiv.org/abs/2108.04204)
* [Universal](https://arxiv.org/abs/1905.03828)


The package provised model classes in the form of Speechbrain Brain classes, that are compatible with the attacks above. Currently implemented:
* Sequence-to-Sequence models with RNNs and Transformers, also supporting the CTC loss and CTC decoding. Compatible with [pretrained](https://huggingface.co/speechbrain/asr-crdnn-rnnlm-librispeech) [speechbrain](https://huggingface.co/speechbrain/asr-crdnn-transformerlm-librispeech) [models](https://huggingface.co/speechbrain/asr-transformer-transformerlm-librispeech).
* RNN-CTC model (DeepSpeech-like) with character tokenizer.
* RNN-Transducer models
* Wav2Vec2 ASR and Pretraining (compatible with pretrained huggingface and fairseq models).

We also provide data preparation and loading functions for the LibriSpeech, CommonVoice and SpeechCommands datasets, based on SpeechBrain recipes.

## Install 
Before installing robust_speech you should have installed PyTorch (>=1.8.0,<=1.10.1) and cuda support (if you want GPU support) in your environment. Testing was conducting with CUDA 9.2 and 10.2 on a Nvidia RTX 2080Ti. Default options assume GPU support; use the `--device=cpu` option in your scripts if do not have it.

```
git clone https://github.com/RaphaelOlivier/robust_speech.git
cd robust_speech
pip install .
```

Additionally, some recipes may require the following lines.

```
# the following lines are required to support [randomized smoothing with ROVER voting](https://arxiv.org/abs/2112.03000):
git clone https://github.com/RaphaelOlivier/SCTK.git
cd SCTK
make config && make all && make check && make install && make doc
export ROVER_PATH=$(echo pwd)/bin/rover # Add this line to your .bash_profile

# the following lines are required to use wav2vec2 models with the fairseq backend. WARNING: Some attacks seem to raise gradient errors with some versions of Pytorch or CUDA. Use the huggingface backend if possible.
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./

```

## Usage
We provide scripts to train and evaluate models (with adversarial attacks in both stages). These scripts are similar to speechbrain training recipes.

We also provide a number of training and evaluation recipes ready for use.

Example

```
# in ./recipes/

# This will download the speechbrain/asr-crdnn-rnnlm-librispeech model from huggingface
python evaluate.py attack_configs/LibriSpeech/pgd/s2s_1000bpe.yaml --root=/path/to/results/folder
```

```
# in ./recipes/

# This will train a model first
python train_model.py train_configs/LibriSpeech/transformer_train.yaml --root=/path/to/results/folder
mv /path/to/training/outputs/folder/*.ckpt /path/to/models/folder/asr-transformer-transformerlm-librispeech/
python evaluate.py attack_configs/LibriSpeech/pgd/trf_5000bpe.yaml --root=/path/to/results/folder --snr=25
```

Some attacks may have trainable parameters. For instance the universal attack computes a perturbation on a training set, then simply applies it at inference time. For these attacks we provide the `fit_attacker.py` script. There should be two attack configuration files: one for fitting the attacker and one for evaluation. The former saves trained parameters with a checkpointer, and the latter loads them with a pretrainer.

```
# in ./recipes/

# This will train a model first
python fit_attacker.py attack_configs/LibriSpeech/universal/ctc_5000bpe_fit.yaml --root=/path/to/results/folder
# Saves parameters in subfolder CKPT+mytimestamp
mv /path/to/training/outputs/folder/*.ckpt /path/to/models/folder/asr-transformer-transformerlm-librispeech/
python evaluate.py attack_configs/LibriSpeech/universal/ctc_5000bpe.yaml --root=/path/to/results/folder --params_subfolder=CKPT+mytimestamp
```

### Computation time
Running strong adversarial attacks on ASR models takes time. The first evaluation script above would run for about 30h on a single RTX 2080Ti. [Speechbrain supports multi-GPU computations](https://speechbrain.readthedocs.io/en/latest/multigpu.html), for instance with the `--data_parallel_backend` option. You may use it with robust_speech if you have many GPUs available, however we have not extensively tested it. 

Alternatively, we recommend that you run all but your final round of experiments using a smaller data split. You can extract a csv yourself (`head -n 101 test-clean.csv > test-clean-short.csv`) and override the csv name in evaluation recipes (`--data_csv_name=test-clean-short`)

### Which recipes are available?
We have provided direct attack recipes for all our attacks on all model architectures currently supported with robust_speech (see [##Features]), and transferred attacks for some of them. However, to run an attack/evaluation recipe the trained model(s) need(s) to be available!

For pretrained Speechbrain Librispeech models the pretrainer will download weights and tokenizers directly from Huggingface and you have nothing to do. So you can run the attack recipes in `s2s_*`, `trf_5000`, `trfctc_5000` and `ctc_5000` directly - as in the first example above.

For the RNN-T and charachter CTC models, there is no pretrained model. You'll have to run the training script first - as in the second example above.

For the Wav2Vec2 models things are slightly trickier. robust_speech can load HuggingFace wav2vec2 models as backend, and these models can be downloaded directly. However, at this point the tokenizers for these models (i.e. the character-label matching) cannot be simply extracted from huggingface and made compatible with robust_speech. Therefore it is necessary to first generate a tokenizer from the data, then do a slight retraining of the final linear layer in wav2vec2. This can be done with `train_configs/LibriSpeech/wav2vec2_fine_tune.py` recipe. One epoch of fine-tuning on the librispeech train-clean-100 split is plenty enough to match the official performance of wav2vec2 base and large models. See the example below
```
# in ./recipes/

# This will train a model first
python train_model.py train_configs/LibriSpeech/wav2vec2_fine_tune.yaml --root=/path/to/results/folder --wav2vec2_hub=facebook/wav2vec2-base-100h
mv /path/to/training/outputs/folder/*.ckpt /path/to/models/folder/wav2vec2-base-100h/
python evaluate.py attack_configs/LibriSpeech/pgd/w2v2_base_100h.yaml --root=/path/to/results/folder --snr=25
```

### Root structure

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
You may change this at will in your custom `.yaml` files or with command line arguments. These folders will be created automatically if the are missing - however, the dataset should already be downloaded in your data_folder. One exception is the light Speech Commands dataset, which will be downloaded automatically if missing.

## Incoming features
* Attacks:
    * [Real-time](https://arxiv.org/abs/2112.07076)
* Data poisoning
* And more!

## Credits
Snippets of code have been copy-pasted from packages [SpeechBrain](https://github.com/speechbrain/speechbrain) (Apache 2.0 license) and [ART](https://github.com/Trusted-AI/adversarial-robustness-toolbox/tree/main/art) (MIT License). This is specified in the files where it happens.

## License
**robust_speech** is released under the Apache License, version 2.0.

## Cite
If you are using robust_speeech for your experiments please cite [this paper](https://arxiv.org/abs/2203.16536):

```bibtex
@article{Olivier2022RI,
  url = {https://arxiv.org/abs/2203.16536},
  author = {Olivier, Raphael and Raj, Bhiksha},
  title = {Recent improvements of ASR models in the face of adversarial attacks},
  journal = {Interspeech},
  year = {2022},  
}

```
