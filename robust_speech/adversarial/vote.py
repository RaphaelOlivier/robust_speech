import os
import subprocess
from typing import List, Tuple
from collections import Counter
import numpy as np
import torch
import tempfile

ROVER_MAX_HYPS=50
ROVER_RECOMMENDED_HYPS=50
TMP_PATH = "/tmp/rover"


class MajorityVote:
    def run(self,asr_outputs:List[np.ndarray],confidence=None, **kwargs) -> np.ndarray: 
        batch_size=len(asr_outputs[0])
        nsamples=len(asr_outputs)
        maj_outputs=[]
        for k in range(batch_size):
            outs=[out[k] for out in asr_outputs]
            scores = [cf[k] for cf in confidence] if confidence else [1] * len(outs)
            max_stc = self.get_max_stc(outs,scores)
            maj_outputs.append(max_stc)
        maj_outputs=np.array(maj_outputs)
        return maj_outputs

    def get_max_stc(self,stc_list,scores):
        unique_sentences = list(set(stc_list))
        stc_idx = {stc:i for i,stc in enumerate(unique_sentences)}
        stc_scores = np.zeros(len(unique_sentences))
        for i,stc in enumerate(stc_list):
            stc_scores[stc_idx[stc]]+=scores[i]
        idx_max = np.argmax(stc_scores)
        stc_max = unique_sentences[idx_max]

        return stc_max

class VoteEnsemble:
    def __init__(self,voter1,voter2, agg_by=50):
        self.voter1=voter1
        self.voter2=voter2 
        self.agg_by=agg_by

    def run(self,asr_outputs, alignments=None, confidence=None,**kwargs):
        num_inputs = len(asr_outputs)
        batch_size = num_inputs // self.agg_by
        num_residuals = num_inputs % self.agg_by
        batch_indices = [range(k,k+batch_size+1) for k in range(0,num_residuals*(batch_size+1),(batch_size+1))] + [range(k,k+batch_size) for k in range(num_residuals*(batch_size+1),num_inputs,batch_size)]
        batches = [[asr_outputs[i] for i in indices] for indices in batch_indices]
        algns_batches = [([alignments[i] for i in indices] if alignments else None) for indices in batch_indices]
        confidence_batches = [([confidence[i] for i in indices] if confidence else None) for indices in batch_indices]
        list_inputs2=[]
        for batch, algns,score in zip(batches,algns_batches,confidence_batches):
            outputs1 = self.voter1.run(batch,alignments=algns, confidence=score)
            list_inputs2.append(outputs1)
        inputs2, alignments2, scores2 = zip(*list_inputs2)
        output2 = self.voter2.run(inputs2, alignments=alignments2, confidence=scores2)
        return output2


class Rover:
    def __init__(self,scheme,exec_path, return_all=False):
        self.rover_path=exec_path
        self.rover_directory = TMP_PATH
        if not os.path.exists(self.rover_directory):
            os.makedirs(self.rover_directory)
        self.outfile = os.path.join(self.rover_directory,'out.txt')
        if scheme=='freq':
            self.rover_options = ['-m',"avgconf", "-a", "1.0", "-c", '0.0']
        elif scheme=='conf':
            self.rover_options = ['-m',"avgconf"]
        else:
            assert scheme=='max'
            self.rover_options = ['-m', 'maxconf']
        self.return_all=return_all
        
    def run(self,asr_outputs, alignments=None, confidence=None, char_alignment=True, **kwargs):
        #print(asr_outputs)
        #print(alignments)
        if len(asr_outputs)>ROVER_MAX_HYPS:
            raise ValueError("ROVER can only handle %d hypothesis at a time but batch contains %d transcriptions. Evaluation will be interrupted."
            %(ROVER_MAX_HYPS,len(asr_outputs)))
        
        if len(asr_outputs)>ROVER_RECOMMENDED_HYPS:
            logger.warn("ROVER is not implemened by default to handle %d hypothesis at a time but batch contains %d transcriptions. Failed instances may occur."
            %(ROVER_RECOMMENDED_HYPS,len(asr_outputs)))
        batch_size=len(asr_outputs[0])
        nsamples=len(asr_outputs)
        final_outputs=[]
        final_alignments=[]
        final_scores=[]
        self.faults=0
        for k in range(batch_size):
            out,align,scores = "",None,None
            outs=[out[k] for out in asr_outputs]
            duration=float(sum([len(stc) for stc in outs]))/nsamples/10
            if duration>0:
                algns = [al[k] for al in alignments] if alignments else [None]*len(outs)
                hypfiles = [self.generate_ctm(stc, i,duration, al,char_alignment=char_alignment) for i,(stc,al) in enumerate(zip(outs,algns))]
                self.run_rover(hypfiles)
                out,align,scores = self.read_ctm(self.outfile)
            if out=="":
                out=self.backup(outs)
                self.faults+=1
                align=None
                scores=None
            final_outputs.append(out)
            final_alignments.append(align)
            final_scores.append(scores)
        
        final_outputs=np.array(final_outputs)
        if self.faults>0:
            print("ROVER failed on %d instances, fall back on majority vote"%self.faults)
        if self.return_all:
            return final_outputs ,final_alignments, final_scores
        return final_outputs

    def generate_ctm(self,sentence, idx, duration, alignments=None, char_alignment=True):
        sentence=sentence.rstrip().lstrip() # remove left and right whitespaces
        sentence=" ".join(filter(lambda w:w!='',sentence.split(' '))) # remove duplicate whitespaces
        lines=[]
        if len(sentence)>0: 
            char_time = duration/len(sentence)
            if alignments is None:
                alignments = np.arange(0., len(sentence), char_time)
            words = sentence.split(' ')
            word_idx=0
            for w in words:
                if char_alignment:
                    start_time=float(alignments[word_idx])
                    end_time=float(alignments[word_idx+len(w)-1])
                    word_time = end_time-start_time
                    line = '0000 A '+str(start_time)+' '+str(word_time)+' '+w +'\n'
                    lines.append(line)
                    word_idx=word_idx+len(w)+1
                else:#word or wordpiece alienment
                    start_time=float(alignments[word_idx])
                    end_time=float(alignments[word_idx+1])
                    word_time = end_time-start_time
                    line = '0000 A '+str(start_time)+' '+str(word_time)+' '+w +'\n'
                    lines.append(line)
                    word_idx=word_idx+1
        hypfile = os.path.join(self.rover_directory,str(idx)+'.txt')
        with open(hypfile,'w') as f:
            for line in lines:
                f.write(line)
        return hypfile

    def run_rover(self,list_hypfiles):
        cmd = [self.rover_path]
        for hypfile in list_hypfiles:
            cmd.append('-h')
            cmd.append(hypfile)
            cmd.append('ctm')
        cmd.append('-o')
        cmd.append(self.outfile)
        cmd=cmd+self.rover_options
        with open(os.path.join(self.rover_directory,"log.txt"),'w') as log:
            subprocess.call(cmd, stdout=subprocess.DEVNULL, stderr=log)
            
    def read_ctm(self,filepath):
        words=[]
        with open(filepath,'r') as f:
            for line in f:
                elts = line.split(' ')
                assert len(elts)==6
                word=elts[4]
                words.append(word)
        sentence=" ".join(words)
        sentence=sentence.upper()
        return sentence, None, None

    def backup(self,outs):
        counts=Counter(outs)
        max_elt,max_count=counts.most_common(1)[0]
        return max_elt
