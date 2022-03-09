import difflib
import torch
import speechbrain as sb 
import robust_speech as rs
from speechbrain.dataio.batch import PaddedBatch 

from robust_speech.adversarial.attacks.attacker import Attacker

ELITE_SIZE=2
TEMPERATURE=0.01
MUTATION_PROB = 0.0005
EPS_NUM_STRIDES = 512

class GeneticAttack(Attacker):

    def __init__(self,asr_brain,nb_iter=100,population_size=20, eps=0.01,targeted=False):
        self.asr_brain=asr_brain
        self.nb_iter=nb_iter
        self.population_size=population_size
        self.eps=eps
        self.targeted=targeted

    def perturb(self,batch):
        if len(batch.sig[0])>1:
            raise NotImplementedError("%s only supports batch size 1"%self.__class__.__name__)
        pop_batch = self._gen_population(batch).to(self.asr_brain.device)

        for idx in range(self.nb_iter):
            pop_scores = self._score(pop_batch)
            _, elite_indices = torch.topk(pop_scores,ELITE_SIZE,largest=True,sorted=True)

            scores_logits = torch.exp(pop_scores / temp)
            pop_probs = scores_logits / torch.sum(scores_logits)
            elite_sig = pop_batch.sig[elite_indices]
            child_sig = self._crossover(pop_batch.sig,pop_probs,self.population_size - ELITE_SIZE)
            child_sig = self._mutation(child_sig)
            pop_batch.sig = torch.stack([elite_sig,child_sig],dim=0)
        return pop_batch.sig[elite_indices[0]].unsqueeze(0).to(batch.sig[0].device)

    def _mutation(self,wavs):
        wav_size = waves.size()
        mutation_mask = torch.rand(*wav_size,device=wavs.device) < MUTATION_PROB
        n_mutations = mutation_mask.sum()
        mutations = torch.tensor(
            np.random.choice(range(-self.eps, self.eps,self.eps/EPS_NUM_STRIDES),size=n_mutations),
            device=wavs.device,
            dtype=wavs.dtype
        )
        wavs = wavs.view(-1)
        wavs[mutation_mask] += mutations
        wavs = wavs.view(wav_size)
        return wavs

    def _gen_population(self,batch):
        # from (1,len) to (self.population_size,len)
        new_wavs = batch.sig[0].expand(self.population_size,batch.sig[0].size(1))
        new_wavs = self._mutation(new_wavs)
        new_lengths = batch.sig[1].expand(self.population_size)
        dic = {"sig":(new_wavs,new_lengths)}
        for k in batch.__dict__:
            if k != "sig":
                val = batch.__dict__[k]
                if isinstance(torch.Tensor,val):
                    new_val = batch.__dict__[k].expand(self.population_size,*val.size()[1:])
                    dic[k]=new_val
        pop_batch = PaddedBatch(dic)
        return pop_batch

    def _score(self,batch):
        predictions = self.asr_brain.compute_forward(batch)
        pred_tokens = self.asr_brain.get_tokens(predictions)
        tokens = batch.tokens
        scores=[]
        for i in range(self.population_size):
            sm = difflib.SequenceMatcher(None,pred_tokens[i],tokens[i])
            score = sm.ratio()
            if not self.targeted:
                score = -score
            scores.append(score)
        return batch.sig[0].new_(scores)
        
    def _crossover(self,wavs,pop_probs,n):
        new_wavs = wavs[np.random.choice(self.population_size, p=pop_probs.detach().cpu().numpy(),size=2*n)]
        wavs1,wavs2 = new_wavs[:n],new_wavs[n:]
        mask = torch.rand(wavs1.size()) < 0.5
        wavs1[mask] = wavs2[mask]
        return wavs1