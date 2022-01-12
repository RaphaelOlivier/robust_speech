import torch
from speechbrain.utils.edit_distance import accumulatable_wer_stats
def snr(audio,perturbation, rel_length):
    length = (audio.size(1)*rel_length).long()
    num = torch.sum(torch.tensor([torch.square(audio[i,:length[i]]).sum() for i in range(audio.size(0))]))
    den = torch.sum(torch.tensor([torch.square(perturbation[i,:length[i]]).sum() for i in range(audio.size(0))]))
    ratio = 10*torch.log10(num/den)
    return torch.round(ratio).long().item()

def wer(str1,str2):
    l1 = str1.split(" ")
    l2 = str2.split(" ")
    stats = accumulatable_wer_stats([l1],[l2])
    return round(stats["WER"],2)
def cer(str1,str2):
    l1 = [c for c in str1]
    l2 = [c for c in str2]
    stats = accumulatable_wer_stats([l1],[l2])
    return round(stats["WER"],2)