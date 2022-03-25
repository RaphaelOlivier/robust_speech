"""
Genetic adversarial attack (https://arxiv.org/abs/1801.00554)
"""

import copy

import numpy as np
import speechbrain as sb
import torch
from speechbrain.dataio.batch import PaddedBatch, PaddedData
from speechbrain.utils.edit_distance import accumulatable_wer_stats

import robust_speech as rs
from robust_speech.adversarial.attacks.attacker import Attacker

ELITE_SIZE = 2
TEMPERATURE = 0.02
MUTATION_PROB = 0.0005
EPS_NUM_STRIDES = 4


class GeneticAttack(Attacker):
    """
    Implementation of the Black-Box genetic adversarial attack for ASR models (https://arxiv.org/abs/1801.00554)
    The original implementation (https://github.com/nesl/adversarial_audio) was slightly changed:
        -untargeted attacks are supported
        -mutations occur in float space rather than byte space (for smoother integration to pytorch)

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
        pop_batch, max_wavs, min_wavs = self._gen_population(batch)

        for idx in range(self.nb_iter):
            pop_scores = self._score(pop_batch)
            _, elite_indices = torch.topk(
                pop_scores, ELITE_SIZE, largest=not self.targeted, sorted=True, dim=-1
            )
            scores_logits = torch.exp((pop_scores) / TEMPERATURE)
            if self.targeted:
                scores_logits = 1.0 - scores_logits
            pop_probs = scores_logits / \
                torch.sum(scores_logits, dim=-1, keepdim=True)
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
            torch.rand(*wav_size, device=wavs.device) < MUTATION_PROB
        ).reshape(-1)
        n_mutations = int(mutation_mask.sum())
        rg_mutations = np.arange(-self.eps, self.eps,
                                 self.eps / EPS_NUM_STRIDES)
        mutations = torch.tensor(
            np.random.choice(rg_mutations, size=n_mutations), device=wavs.device, dtype=wavs.dtype
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
        scores = scores / scores.max(dim=1, keepdim=True)[0]
        return scores

    def _crossover(self, batches, pop_probs, num_crossovers):
        # pop_probs : (batch_size x pop_size)
        batch_size = pop_probs.size(0)
        new_wavs_1 = []
        new_wavs_2 = []
        for i in range(batch_size):
            rg_crossover = np.random.choice(
                self.population_size, p=pop_probs[i].detach().cpu().numpy(), size=2 * num_crossovers
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
