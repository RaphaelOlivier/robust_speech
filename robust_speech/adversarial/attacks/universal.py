"""
Variations of the Unviersal attack (https://arxiv.org/pdf/1905.03828.pdf)
"""


import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from robust_speech.models.seq2seq import S2SASR
from robust_speech.models.ctc import CTCASR

import robust_speech as rs
from robust_speech.adversarial.attacks.attacker import Attacker
from robust_speech.adversarial.utils import (
    l2_clamp_or_normalize,
    linf_clamp,
    rand_assign,
)


def reverse_bound_from_rel_bound(batch, rel, order=np.inf):
    """From a relative eps bound, reconstruct the absolute bound for the given batch"""
    wavs, wav_lens = batch.sig
    wav_lens = [int(wavs.size(1) * r) for r in wav_lens]
    epss = []
    for i in range(len(wavs)):
        eps = torch.norm(wavs[i, : wav_lens[i]], p=order) / rel
        epss.append(eps)
    return torch.tensor(epss).to(wavs.device)

class UniversalAttack(Attacker):
    """
    Implementation of the Universal attack (https://arxiv.org/pdf/1905.03828.pdf)
    The attack performs nb_iter steps of size eps_iter, while always staying
    within eps from the initial point.

    Arguments
    ---------
    asr_brain: rs.adversarial.brain.ASRBrain
       brain object.
    snr: float
       maximum distortion.
    eps: float
       maximum distortion.
    nb_iter: int
       number of iterations.
    eps_iter: float
       attack step size.
    rand_init: (optional bool)
       random initialization.
    clip_min: (optional) float
       mininum value per input dimension.
    clip_max: (optional) float
       maximum value per input dimension.
    order: (optional) int
       the order of maximum distortion (inf or 2).
    targeted: bool
       if the attack is targeted.
    train_mode_for_backward: bool
       whether to force training mode in backward passes (necessary for RNN models)
    """

    def __init__(
        self,
        asr_brain,
        snr=5,
        eps=0.3,
        nb_iter=40,
        rel_eps_iter=5.,
        rand_init=True,
        clip_min=None,
        clip_max=None,
        order=np.inf,
        l1_sparsity=None,
        targeted=False,
        train_mode_for_backward=True,
    ):

        self.clip_min = clip_min if clip_min is not None else -10
        self.clip_max = clip_max if clip_max is not None else 10
        self.eps = eps
        self.nb_iter = nb_iter
        self.rel_eps_iter = rel_eps_iter
        self.rand_init = rand_init
        self.order = order
        self.targeted = targeted
        self.asr_brain = asr_brain
        self.l1_sparsity = l1_sparsity
        self.train_mode_for_backward = train_mode_for_backward

        assert isinstance(snr, int)
        self.rel_eps = torch.pow(torch.tensor(10.0), float(snr) / 20)

        assert isinstance(self.rel_eps_iter, torch.Tensor) or isinstance(
            self.rel_eps_iter, float
        )
        assert isinstance(self.eps, torch.Tensor) or isinstance(self.eps, float)

    def compute_universal_perturbation(self, loader):
        if isinstance(self.asr_brain,S2SASR):
            decode = self.asr_brain.tokenizer.decode_ids
        elif isinstance(self.asr_brain,CTCASR):
            decode = self.asr_brain.tokenizer.decode_ids
        else:
            raise NotImplementedError
        
        if isinstance(self.rel_eps_iter, torch.Tensor):
            assert self.eps_iter.dim() == 1
            eps_iter = self.rel_eps_iter.unsqueeze(1).to(self.asr_brain.device)
        else:
            eps_iter = torch.Tensor([self.rel_eps_iter]).to(self.asr_brain.device)

        self.asr_brain.module_train()

        delta = None
        success_rate = 0
        
        epoch = 0
        while success_rate < 0.8:
            print(f'{epoch}s epoch')
            epoch+=1
            print("GENERATE UNIVERSAL PERTURBATION")
            ### GENERATE CANDIDATE FOR UNIVERSAL PERTURBATION
            for idx, batch in enumerate(tqdm(loader, dynamic_ncols=True)):
                batch = batch.to(self.asr_brain.device)
                wav_init, wav_lens = batch.sig

                if delta is None:
                    delta = torch.zeros_like(wav_init)
                
                # if idx == 20:
                #     break
                #     # raise NotImplementedError
            

                # Slice or Pad to match the shape with data point x
                if wav_init.shape[1] <= delta.shape[1]:
                    delta_x = torch.zeros_like(wav_init)
                    delta_x[:,:delta.shape[1]] = delta[:,:wav_init.shape[1]].detach()
                else:
                    delta_x = torch.zeros_like(wav_init)
                    delta_x[:,:delta.shape[1]] = delta.detach()

                _,_,predicted_tokens_origin = self.asr_brain.compute_forward(batch, rs.Stage.ADVTRUTH)
                predicted_words_origin = [
                    decode(utt_seq).split(" ")
                    for utt_seq in predicted_tokens_origin
                ]

                r = torch.rand_like(delta_x) / 1e+4
                r.requires_grad_()

                batch.sig = wav_init + delta_x, wav_lens
                _,_,predicted_tokens_adv = self.asr_brain.compute_forward(batch, rs.Stage.ADVTARGET)
                predicted_words_adv = [
                    decode(utt_seq).split(" ")
                    for utt_seq in predicted_tokens_adv
                ]

                # self.asr_brain.cer_metric.append(batch.id, predicted_words_adv, predicted_words_origin)
                self.asr_brain.cer_metric.append(batch.id, predicted_words_origin, predicted_words_adv)
                CER = self.asr_brain.cer_metric.summarize("error_rate")
                self.asr_brain.cer_metric.clear()
                # print(CER)

                while CER < 50.:
                    batch.sig = wav_init + delta_x + r, wav_lens
                    predictions = self.asr_brain.compute_forward(batch, rs.Stage.ATTACK)
                    # loss = 0.5 * r.norm(dim=1, p=2) - self.asr_brain.compute_objectives(predictions, batch, rs.Stage.ATTACK)
                    ctc = -self.asr_brain.compute_objectives(predictions, batch, rs.Stage.ATTACK) 
                    l2_norm = r.norm(dim=1, p=2).to(self.asr_brain.device)
                    loss = 0.5 * l2_norm + ctc
                    # loss = ctc
                    loss.backward()
                    # print(l2_norm,ctc,CER)
                    grad_sign = r.grad.data.sign()
                    r.data = r.data - 0.001 * grad_sign
                    # r.data = r.data - 0.1 * r.grad.data
                    r.data = linf_clamp(delta_x.data + r.data, self.eps) - delta_x.data
                    
                    # print("delta's mean : ", torch.mean(delta_x).data)
                    # print("r's mean : ",torch.mean(r).data)
                    r.grad.data.zero_()

                    _,_,predicted_tokens_adv = self.asr_brain.compute_forward(batch, rs.Stage.ADVTARGET)
                    predicted_words_adv = [
                        decode(utt_seq).split(" ")
                        for utt_seq in predicted_tokens_adv
                    ]

                    self.asr_brain.cer_metric.append(batch.id, predicted_words_origin, predicted_words_adv)
                    CER = self.asr_brain.cer_metric.summarize("error_rate")
                    self.asr_brain.cer_metric.clear()
                    # print(CER)
                
                # print(f'CER = {CER}')
                delta_x = linf_clamp(delta_x + r.data, self.eps)

                if delta.shape[1] <= delta_x.shape[1]:
                    delta = delta_x[:,:delta.shape[1]].detach()
                else:
                    delta = torch.zeros_like(delta)
                    delta[:,:delta_x.shape[1]] = delta_x.detach()

            # print(f'MAX OF INPUT WAVE IS {torch.max(wav_init).data}')
            # print(f'AVG OF INPUT WAVE IS {torch.mean(wav_init).data}')
            # print(f'MAX OF DELTA IS {torch.max(delta).data}')
            # print(f'AVG OF DELTA IS {torch.mean(delta).data}')
            print('CHECK SUCCESS RATE OVER ALL TRAIING SAMPLES')
            ### TO CHECK SUCCESS RATE OVER ALL TRAINING SAMPLES
            total_sample = 0.
            fooled_sample = 0.

            for idx, batch in enumerate(tqdm(loader, dynamic_ncols=True)):
                batch = batch.to(self.asr_brain.device)
                wav_init, wav_lens = batch.sig
                if wav_init.shape[1] <= delta.shape[1]:
                    delta_x = torch.zeros_like(wav_init)
                    delta_x[:,:delta.shape[1]] = delta[:,:wav_init.shape[1]]
                else:
                    delta_x = torch.zeros_like(wav_init)
                    delta_x[:,:delta.shape[1]] = delta[:,:wav_init.shape[1]]
                # if idx == 400:
                #     break
                #     raise NotImplementedError

                ### CER(Xi)
                _,_,predicted_tokens_origin = self.asr_brain.compute_forward(batch, rs.Stage.ADVTRUTH)
                
                predicted_words_origin = [
                    decode(utt_seq).split(" ")
                    for utt_seq in predicted_tokens_origin
                ]

                ### CER(Xi + v)
                batch.sig = wav_init + delta_x, wav_lens
                _,_,predicted_tokens_adv = self.asr_brain.compute_forward(batch, rs.Stage.ADVTRUTH)
                predicted_words_adv = [
                    decode(utt_seq).split(" ")
                    for utt_seq in predicted_tokens_adv
                ]
                self.asr_brain.cer_metric.append(batch.id, predicted_words_origin, predicted_words_adv)
                CER = self.asr_brain.cer_metric.summarize("error_rate")
                self.asr_brain.cer_metric.clear()

                total_sample += 1.
                if CER > 50.:
                    fooled_sample += 1.

            success_rate = fooled_sample / total_sample
            print(f'SUCCESS RATE IS {success_rate}')

        self.univ_perturb = delta.detach()

 
    def perturb(self, batch):
        """
        Compute an adversarial perturbation
        Arguments
        ---------
        batch : sb.PaddedBatch
           The input batch to perturb

        Returns
        -------
        the tensor of the perturbed batch
        """
        if self.train_mode_for_backward:
            self.asr_brain.module_train()
        else:
            self.asr_brain.module_eval()

        save_device = batch.sig[0].device
        batch = batch.to(self.asr_brain.device)
        save_input = batch.sig[0]
        wav_init = torch.clone(save_input)

        if wav_init.shape[1] <= self.univ_perturb.shape[1]:
            delta = self.univ_perturb[:,:wav_init.shape[1]]
        else:
            delta = torch.zeros_like(wav_init)
            delta[:,:self.univ_perturb.shape[1]] = self.univ_perturb

        wav_adv = wav_init + delta
        # self.eps = 1.0
        batch.sig = save_input, batch.sig[1]
        batch = batch.to(save_device)
        self.asr_brain.module_eval()
        return wav_adv.data.to(save_device)

