
import os
import sys
import torch
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main
from advertorch.attacks import L2PGDAttack
from robust_speech.adversarial.attacks.pgd import ASRL2PGDAttack
from robust_speech.adversarial.metrics import snr, wer, cer
from robust_speech.adversarial.brain import ASRBrain
from robust_speech.utils import make_batch_from_waveform, transcribe_batch, load_audio
import robust_speech as rs
# Define training procedure
class ASR(ASRBrain):

    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        if not stage == rs.Stage.ATTACK:
            batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        tokens_bos, _ = batch.tokens_bos
        #wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
        # Add augmentation if specified
        if stage == sb.Stage.TRAIN:
            if hasattr(self.modules, "env_corrupt"):
                wavs_noise = self.modules.env_corrupt(wavs, wav_lens)
                wavs = torch.cat([wavs, wavs_noise], dim=0)
                wav_lens = torch.cat([wav_lens, wav_lens])
                tokens_bos = torch.cat([tokens_bos, tokens_bos], dim=0)

            if hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs, wav_lens)
        # Forward pass
        feats = self.modules.wav2vec2(wavs)
        if stage == rs.Stage.ATTACK:
            x = self.modules.enc(feats)
        else:
            x = self.modules.enc(feats.detach())
         # Compute outputs
        p_tokens = None
        logits = self.modules.ctc_lin(x)
        p_ctc = self.hparams.log_softmax(logits)
        
        if stage not in [sb.Stage.TRAIN, rs.Stage.ATTACK] :
            p_tokens = sb.decoders.ctc_greedy_decode(
                p_ctc, wav_lens, blank_id=self.hparams.blank_index
            )
        return p_ctc, wav_lens, p_tokens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (CTC+NLL) given predictions and targets."""

        p_ctc, wav_lens, predicted_tokens = predictions

        ids = batch.id
        tokens_eos, tokens_eos_lens = batch.tokens_eos
        tokens, tokens_lens = batch.tokens
        
        if hasattr(self.modules, "env_corrupt") and stage == sb.Stage.TRAIN:
            tokens_eos = torch.cat([tokens_eos, tokens_eos], dim=0)
            tokens_eos_lens = torch.cat(
                [tokens_eos_lens, tokens_eos_lens], dim=0
            )
            tokens = torch.cat([tokens, tokens], dim=0)
            tokens_lens = torch.cat([tokens_lens, tokens_lens], dim=0)
        
        loss_ctc = self.hparams.ctc_cost(p_ctc, tokens, wav_lens, tokens_lens)
        loss = loss_ctc

        if stage != sb.Stage.TRAIN and stage != rs.Stage.ATTACK:
            # Decode token terms to words
            predicted_words = [
                self.tokenizer.decode_ndim(utt_seq).split(" ")
                for utt_seq in predicted_tokens
            ]
            target_words = [wrd.split(" ") for wrd in batch.wrd]
            self.wer_metric.append(ids, predicted_words, target_words)
            self.cer_metric.append(ids, predicted_words, target_words)

        return loss

if __name__ == "__main__":

    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    
    audio_file = hparams["attack_input_file"]
    waveform = load_audio(audio_file, hparams)
        # Fake a batch:
    wavs = waveform.unsqueeze(0)
    rel_length = torch.tensor([1.0])
    run_on_main(hparams["pretrainer"].collect_files)
    hparams["pretrainer"].load_collected(device=run_opts["device"])

    asr_brain = ASR(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    asr_brain.tokenizer = hparams["tokenizer"]

    batch = make_batch_from_waveform(waveform,"", [], hparams)
    asr_brain.modules.eval()
    words, tokens = transcribe_batch(asr_brain,batch)
    print(words)

    if hparams["attack_target"] is not None:
        #assert hparams["attack_class"]["targeted"], "Target cannot be specified for untargeted attack"
        words = hparams["attack_target"]
        tokens = hparams["tokenizer"].encode_sequence(words)
        print(words)
        
    asr_brain.modules.train()
    batch = make_batch_from_waveform(waveform,words, tokens, hparams)
    attack_class = hparams["attack_class"]
    attack = attack_class(asr_brain)
    adv_wavs = attack.perturb(batch)
    batch.sig=adv_wavs,batch.sig[1]
    asr_brain.modules.eval()
    adv_words, adv_tokens = transcribe_batch(
            asr_brain, batch
        )
    print(adv_words)
    
    print(snr(wavs,wavs-adv_wavs,rel_length))
    print(wer(words,adv_words))
    print(cer(words,adv_words))

