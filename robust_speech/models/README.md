# robust_speech models

Brain classes for various ASR architectures supported by robust-speech. Note that the inference `sb.pretrained.EncoderDecoderASR` class is not recommended with robust-speech: to evaluate under attacks all models should be nested in a brain class. To load parameters from local or HuggingFace paths one must use the speechbrain `Pretrainer`. This is handled in our evaluation script.

These classes are common architectures, often corresponding to Speechbrain pretrained models on LibriSpeech. They are compatible with adversarial attacks and our data loading functions. We integrated brain classes and data loading functions to the package for simplicity; in Speechbrain, such code is typically implemented directly in the recipes.

Adding more brain classes is typically straightforward as it is Speechbrain. The only additional constraint in robust_speech is to support the rs.Stage.ATTACK stage, which should maintain a computation graph from the input.