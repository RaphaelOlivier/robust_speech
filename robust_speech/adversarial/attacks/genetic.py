import copy
import numpy as np
import torch
import speechbrain as sb 
import robust_speech as rs
from speechbrain.dataio.batch import PaddedBatch, PaddedData
from speechbrain.utils.edit_distance import accumulatable_wer_stats

from robust_speech.adversarial.attacks.attacker import Attacker

ELITE_SIZE=2
TEMPERATURE=10.
MUTATION_PROB = 0.001
EPS_NUM_STRIDES = 128

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
    def __init__(self,asr_brain,nb_iter=100,population_size=5, eps=0.01,targeted=False):
        self.asr_brain=asr_brain
        self.nb_iter=nb_iter
        self.population_size=population_size
        self.eps=eps
        self.targeted=targeted

    def perturb(self,batch):
        if len(batch.sig[0])>1:
            raise NotImplementedError("%s only supports batch size 1"%self.__class__.__name__)
        pop_batch, max_wavs, min_wavs = self._gen_population(batch)
        pop_batch=pop_batch.to(self.asr_brain.device)

        for idx in range(self.nb_iter):
            pop_scores = self._score(pop_batch)
            _, elite_indices = torch.topk(pop_scores,ELITE_SIZE,largest=not self.targeted,sorted=True)
            scores_logits = torch.exp((pop_scores - pop_scores.min()) / TEMPERATURE)
            if self.targeted:
                scores_logits = - scores_logits + scores_logits.max()
            pop_probs = scores_logits / torch.sum(scores_logits)
            #print(pop_scores,scores_logits,pop_probs)
            elite_sig = pop_batch.sig[0][elite_indices]
            child_sig = self._crossover(pop_batch.sig[0],pop_probs,self.population_size - ELITE_SIZE)
            child_sig = self._mutation(child_sig)
            pop_batch.sig = torch.clamp(torch.cat([elite_sig,child_sig],dim=0),min=min_wavs,max=max_wavs), pop_batch.sig[1]
        return pop_batch.sig[0][elite_indices[0]].unsqueeze(0).to(batch.sig[0].device)

    def _mutation(self,wavs):
        wav_size = wavs.size()
        mutation_mask = (torch.rand(*wav_size,device=wavs.device) < MUTATION_PROB).reshape(-1)
        n_mutations = int(mutation_mask.sum())
        rg = np.arange(-self.eps, self.eps,self.eps/EPS_NUM_STRIDES)
        mutations = torch.tensor(
            np.random.choice(rg,size=n_mutations),
            device=wavs.device,
            dtype=wavs.dtype
        )
        wavs = wavs.reshape(-1)
        wavs[mutation_mask] += mutations
        wavs = wavs.reshape(wav_size)
        return wavs

    def _gen_population(self,batch):
        # from (1,len) to (self.population_size,len)
        new_wavs = batch.sig[0].expand(self.population_size,batch.sig[0].size(1))
        max_wavs = torch.clone(new_wavs) + self.eps
        min_wavs = max_wavs - 2*self.eps
        new_wavs = self._mutation(new_wavs)
        new_lengths = batch.sig[1].expand(self.population_size)

        # new batch

        keys = [k for k in dir(batch) if not (k.startswith("_") or hasattr(PaddedBatch,k))]
        padded_keys = [k for k in keys if isinstance(batch[k],PaddedData)]
        dic = {k:(batch[k][0][0] if k in padded_keys else batch[k][0]) for k in keys}
        dics = [copy.deepcopy(dic) for _ in range(self.population_size)]
        for i in range(self.population_size):
            dics[i]["sig"] = new_wavs[i]
        pop_batch = PaddedBatch(dics,padded_keys=padded_keys)
        return pop_batch, max_wavs, min_wavs

    def _score(self,batch):
        predictions = self.asr_brain.compute_forward(batch, stage=rs.Stage.ATTACK)
        loss = self.asr_brain.compute_objectives(predictions, batch, stage=rs.Stage.ATTACK, reduction="none")
        tokens = batch.tokens[0]
        scores=loss

        return scores

    def _crossover(self,wavs,pop_probs,n):
        rg = np.random.choice(self.population_size, p=pop_probs.detach().cpu().numpy(),size=2*n)
        new_wavs = wavs[rg]
        wavs1,wavs2 = new_wavs[:n],new_wavs[n:]
        mask = torch.rand(wavs1.size()) < 0.5
        wavs1[mask] = wavs2[mask]
        return wavs1