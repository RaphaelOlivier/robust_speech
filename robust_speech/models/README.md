# robust_speech models

Brain classes for various ASR architectures supported by robust-speech.

These classes are common architectures, often corresponding to Speechbrain pretrained models on LibriSpeech. We integrated them to the package for simplicity; in Speechbrain, such brain classes are typically implemented directly in the recipes.

Adding more brain classes is as straightforward as it is Speechbrain. The only additional constraint in robust_speech is to support the rs.Stage.ATTACK stage, which should maintain a computation graph from the input.