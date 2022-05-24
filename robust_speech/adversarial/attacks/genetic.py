"""
Genetic adversarial attack 
Based on https://arxiv.org/abs/1801.00554
Enhanced with the momentum mutation from https://arxiv.org/pdf/1805.07820.pdf
"""

import copy

import numpy as np
import speechbrain as sb
import torch
import torch.nn.functional as F
from speechbrain.dataio.batch import PaddedBatch, PaddedData
from speechbrain.utils.edit_distance import accumulatable_wer_stats

import robust_speech as rs
from robust_speech.adversarial.attacks.attacker import Attacker

ELITE_SIZE = 2
TEMPERATURE = 0.02
MUTATION_PROB_INIT = 0.0005
EPS_NUM_STRIDES = 4
ALPHA_MOMENTUM = 0.99
EPS_MOMENTUM = 0.001


class GeneticAttack(Attacker):
    """
    Implementation of the Black-Box genetic adversarial attack for ASR models
    (https://arxiv.org/abs/1801.00554)
    The original implementation (https://github.com/nesl/adversarial_audio)
     was slightly changed:
        -untargeted attacks are supported
        -mutations occur in float space rather than byte space
        (for smoother integration to pytorch)

    Arguments
    ---------
     asr_brain: rs.adversarial.brain.ASRBrain
        brain object.
     nb_iter: int
        number of iterations.
     population_size: int
        size of the maintained population.
     eps: float
        maximum Linf distortion.
    """

    def __init__(
        self, asr_brain, nb_iter=100, population_size=10, eps=0.01, targeted=False
    ):
        self.asr_brain = asr_brain
        self.nb_iter = nb_iter
        self.population_size = population_size
        self.eps = eps
        self.targeted = targeted

    def perturb(self, batch):
        self.mutation_prob = MUTATION_PROB_INIT
        self.prev_score = None
        pop_batch, max_wavs, min_wavs = self._gen_population(batch)

        for _ in range(self.nb_iter):
            pop_scores = self._score(pop_batch)
            self._momentum_mutation_prob(pop_scores)
            _, elite_indices = torch.topk(
                pop_scores, ELITE_SIZE, largest=True, sorted=True, dim=-1
            )
            pop_probs = F.softmax((pop_scores) / TEMPERATURE, dim=-1)
            elite_sig = self._extract_elite(pop_batch, elite_indices)
            child_sig = self._crossover(
                pop_batch, pop_probs, self.population_size - ELITE_SIZE
            )
            child_sig = self._mutation(child_sig)
            pop_batch = self._update_pop(
                pop_batch, elite_sig, child_sig, min_wavs, max_wavs
            )

        wav_adv = self._extract_best(
            pop_batch, elite_indices).to(batch.sig[0].device)
        return wav_adv

    def _momentum_mutation_prob(self, pop_scores):
        new_score = pop_scores.mean()
        if self.prev_score is not None:
            coeff = abs(new_score-self.prev_score)
            self.mutation_prob = ALPHA_MOMENTUM * self.mutation_prob + \
                (1-ALPHA_MOMENTUM)*EPS_MOMENTUM/max(EPS_MOMENTUM, coeff)
        self.prev_score = new_score

    def _update_pop(self, batches, elite_sig, child_sig, min_wavs, max_wavs):
        # elite_sig : batch_size * elite_size
        # child_sig : batch_size * (pop_size - elite_size)
        pop_sig = torch.clamp(
            torch.cat([elite_sig, child_sig], dim=1), min=min_wavs, max=max_wavs
        ).transpose(0, 1)
        for i, pop_batch in enumerate(batches):
            pop_batch.sig = pop_sig[i], pop_batch.sig[1]
        return batches

    def _extract_best(self, batches, elite_indices):
        sig = []
        for i in range(len(elite_indices)):
            wav = batches[elite_indices[i, 0]].sig[0][i]
            sig.append(wav)
        return torch.stack(sig, 0)

    def _extract_elite(self, batches, elite_indices):
        # elite_indices : (batch_size x elite_size)
        batch_size = elite_indices.size(0)
        sigs = [[] for _ in range(batch_size)]
        for i in range(batch_size):
            for j in range(ELITE_SIZE):
                wav = batches[elite_indices[i][j]].sig[0][i]
                sigs[i].append(wav)
        sigs = [torch.stack(sig, 0) for sig in sigs]
        sigs = torch.stack(sigs, 0)  # (batch_size x elite_size)
        return sigs

    def _mutation(self, wavs):
        wav_size = wavs.size()
        mutation_mask = (
            torch.rand(*wav_size, device=wavs.device) < self.mutation_prob
        ).reshape(-1)
        n_mutations = int(mutation_mask.sum())
        mutations = torch.tensor(
            np.random.normal(scale=self.eps, size=n_mutations),
            device=wavs.device,
            dtype=wavs.dtype,
        )
        wavs = wavs.reshape(-1)
        wavs[mutation_mask] += mutations
        wavs = wavs.reshape(wav_size)
        return wavs

    def _gen_population(self, batch):
        # from (1,len) to (self.population_size,len)
        size = batch.sig[0].size()
        new_wavs = batch.sig[0].unsqueeze(
            0).expand(self.population_size, *size)
        max_wavs = torch.clone(new_wavs).transpose(0, 1) + self.eps
        min_wavs = max_wavs - 2 * self.eps
        new_wavs = self._mutation(new_wavs)

        # new batch
        pop_batches = []
        for i in range(self.population_size):
            pop_batch = copy.deepcopy(batch)
            pop_batch.sig = new_wavs[i], pop_batch.sig[1]
            pop_batches.append(pop_batch.to(self.asr_brain.device))
        return pop_batches, max_wavs, min_wavs

    def _score(self, batches):
        scores = []
        for i in range(self.population_size):
            predictions = self.asr_brain.compute_forward(
                batches[i], stage=rs.Stage.ATTACK
            )
            loss = self.asr_brain.compute_objectives(
                predictions, batches[i], stage=rs.Stage.ATTACK, reduction="batch"
            )
            scores.append(loss.detach())
        scores = torch.stack(scores, dim=1)  # (batch_size x pop_size)
        if self.targeted:
            scores = -scores
        return scores

    def _crossover(self, batches, pop_probs, num_crossovers):
        # pop_probs : (batch_size x pop_size)
        batch_size = pop_probs.size(0)
        new_wavs_1 = []
        new_wavs_2 = []
        for i in range(batch_size):
            rg_crossover = np.random.choice(
                self.population_size,
                p=pop_probs[i].detach().cpu().numpy(),
                size=2 * num_crossovers,
            )
            new_wavs_1_i = [batches[k].sig[0][i]
                            for k in rg_crossover[:num_crossovers]]
            new_wavs_1.append(torch.stack(new_wavs_1_i, 0))
            new_wavs_2_i = [batches[k].sig[0][i]
                            for k in rg_crossover[num_crossovers:]]
            new_wavs_2.append(torch.stack(new_wavs_2_i, 0))
        # (batch_size x num_crossovers x ...)
        new_wavs_1 = torch.stack(new_wavs_1, 0)
        # (batch_size x num_crossovers x ...)
        new_wavs_2 = torch.stack(new_wavs_2, 0)
        mask = torch.rand(new_wavs_1.size()) < 0.5
        new_wavs_1[mask] = new_wavs_2[mask]

        return new_wavs_1
