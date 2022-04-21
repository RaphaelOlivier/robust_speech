
from collections import defaultdict, Counter
import speechbrain as sb
import torch
import torch.nn.functional as F
from speechbrain.decoders.ctc import filter_ctc_output
import robust_speech as rs


class CTCGreedyDecode(sb.decoders.seq2seq.S2SBaseSearcher):
    """Binding between Seq2Seq models and CTC decoders"""

    def __init__(self, blank_index, ctc_lin, log_softmax):
        super(CTCGreedyDecode, self).__init__(None, None, None, None)
        self.blank_index = blank_index
        self.ctc_lin = ctc_lin
        self.log_softmax = log_softmax

    def forward(self, enc_states, wav_len):
        logits = self.ctc_lin(enc_states)
        p_ctc = self.log_softmax(logits)
        p_tokens = sb.decoders.ctc_greedy_decode(
            p_ctc, wav_len, blank_id=self.blank_index
        )
        return p_tokens, None


def prefix_beam_search(ctc, blank_id, lm=None, k=25, alpha=0.30, beta=5, prune=0.001):
    W = lambda l: re.findall(r'\w+[\s|>]', l)
    lm = (lambda l: 1) if lm is None else lm # if no LM is provided, just set to function returning 1
    F = ctc.shape[1]
    ctc = torch.vstack((ctc.new_zeros(F), ctc)) # just add an imaginative zero'th step (will make indexing more intuitive)
    T = ctc.shape[0]
    alphabet = torch.arange(F)
    # STEP 1: Initiliazation
    O = torch.tensor([]).long()
    Pb, Pnb = defaultdict(Counter), defaultdict(Counter)
    Pb[0][tuple(O.tolist())] = 1
    Pnb[0][tuple(O.tolist())] = 0
    A_prev = [tuple(O.tolist())]
    # END: STEP 1

    # STEP 2: Iterations and pruning
    for t in range(1, T):
        #print(A_prev)
        for l in A_prev:
            l_t=torch.tensor(l).long()
            if alpha>0:
                lm_prob = lm(l_t.to(ctc.device))[-1]
                probs = alpha * torch.exp(lm_prob) + (1.-alpha)*ctc[t]
            else:
                probs=ctc[t]
            pruned_alphabet = [alphabet[i] for i in torch.where(probs > prune)[0]]
            for c in pruned_alphabet:
                c_ix = c
                # END: STEP 2
                # STEP 3: “Extending” with a blank
                # END: STEP 3
                
                # STEP 4: Extending with the end character
                l_plus_t = torch.cat([l_t,c.unsqueeze(0)])
                l_plus = tuple(l_plus_t.tolist())
                if len(l) > 0 and c == l[-1]:
                    Pnb[t][l_plus] += ctc[t][c_ix] * Pb[t - 1][l]
                    Pnb[t][l] += ctc[t][c_ix] * Pnb[t - 1][l]
            # END: STEP 4

                # STEP 5: Extending with any other non-blank character and LM constraints
                elif len(l) > 0 and c==blank_id:
                    Pnb[t][l_plus] += probs[c_ix] * (Pb[t - 1][l] + Pnb[t - 1][l])
                else:
                    Pnb[t][l_plus] += ctc[t][c_ix] * (Pb[t - 1][l] + Pnb[t - 1][l])
                    #print(Pnb[t])
                # END: STEP 5

                # STEP 6: Make use of discarded prefixes
                if l_plus not in A_prev:
                    Pb[t][l_plus] += ctc[t][-1].item() * (Pb[t - 1][l_plus] + Pnb[t - 1][l_plus])
                    Pnb[t][l_plus] += ctc[t][c_ix].item() * Pnb[t - 1][l_plus]
                # END: STEP 6

        # STEP 7: Select most probable prefixes
        A_next = Pb[t] + Pnb[t]
        sorter = lambda l: A_next[l] #* (len(l) + 1) ** beta
        A_prev = sorted([t for t in A_next], key=sorter, reverse=True)[:k]
        # END: STEP 7
    
    return [token for token in A_prev[0] if token != blank_id]

class CTCBeamLMDecode(sb.decoders.seq2seq.S2SBaseSearcher):
    """Binding between Seq2Seq models and CTC decoders"""
    def __init__(self, blank_index, ctc_lin, log_softmax, beam_size=1, lm=None, lm_weight=0.):
        super(CTCBeamLMDecode, self).__init__(None, None, None, None)
        self.blank_index = blank_index
        self.ctc_lin = ctc_lin
        self.log_softmax = log_softmax
        self.lm=lm 
        self.lm_weight=lm_weight
        self.beam_size=beam_size

    def forward(self, enc_states, wav_len):
        logits = self.ctc_lin(enc_states)
        p_ctc = torch.exp(self.log_softmax(logits))
        p_tokens=[]
        for i in range(len(wav_len)):
            ctc = p_ctc[i,:int(wav_len[i]*p_ctc.size(1))]
            tokens = prefix_beam_search(
                ctc, lm=self.lm, blank_id=self.blank_index, k=self.beam_size, alpha=self.lm_weight, prune=0.001
            )
            p_tokens.append(tokens)
        return p_tokens, None


def prefix_greedy_search(ctc, blank_id, lm=None, alpha=0.30, temperature_lm=1.0):
    T = ctc.shape[0]
    # STEP 1: Initiliazation
    l = torch.tensor([1],device=ctc.device).long()
    filt_l = l
    # STEP 2: Iterations and pruning
    for t in range(T):
        
        if alpha>0 and len(filt_l)>0:
            lm_log_prob = lm(filt_l)[-1]
            lm_prob = F.softmax(lm_log_prob / temperature_lm,dim=-1)
            probs = alpha * lm_prob + (1.-alpha)*ctc[t]
        else:
            probs=ctc[t]
        next_c = probs.argmax()
        l = torch.cat([l,next_c.unsqueeze(0)])
        filt_l = ctc.new(filter_ctc_output(l.tolist(),blank_id)).long()
    res = filt_l.tolist()[1:]
    return res

class CTCGreedyLMDecode(sb.decoders.seq2seq.S2SBaseSearcher):
    """Binding between Seq2Seq models and CTC decoders"""
    def __init__(self, blank_index, ctc_lin, log_softmax, lm=None, lm_weight=0.):
        super(CTCGreedyLMDecode, self).__init__(None, None, None, None)
        self.blank_index = blank_index
        self.ctc_lin = ctc_lin
        self.log_softmax = log_softmax
        self.lm=lm 
        self.lm_weight=lm_weight

    def forward(self, enc_states, wav_len):
        logits = self.ctc_lin(enc_states)
        p_ctc = torch.exp(self.log_softmax(logits))
        p_tokens=[]
        for i in range(len(wav_len)):
            ctc = p_ctc[i,:int(wav_len[i]*p_ctc.size(1))]
            tokens = prefix_greedy_search(
                ctc, lm=self.lm, blank_id=self.blank_index, alpha=self.lm_weight
            )
            p_tokens.append(tokens)
        return p_tokens, None