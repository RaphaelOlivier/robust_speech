import torch
import torchaudio
from speechbrain.utils.edit_distance import accumulatable_wer_stats

def snr(audio,perturbation, rel_length=torch.tensor([1.])):
    length = (audio.size(1)*rel_length).long()
    num = torch.tensor([torch.square(audio[i,:length[i]]).sum() for i in range(audio.size(0))])
    den = torch.tensor([torch.square(perturbation[i,:length[i]]).sum() for i in range(audio.size(0))])
    ratio = 10*torch.log10(num/den)
    return torch.round(ratio).long()

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

from speechbrain.utils.metric_stats import MetricStats

class SNRComputer(MetricStats):
    """Tracks Signal to Noise Ratio
    """
    def __init__(self,**kwargs):

        def metric(batch, adv_wav):
            return snr(batch.sig[0],adv_wav-batch.sig[0],batch.sig[1])
        super().__init__(metric,**kwargs)

class AudioSaver:
    """Saves adversarial audio files
    """
    def __init__(self,save_audio_path, sample_rate = 16000):
        self.save_audio_path=save_audio_path
        self.sample_rate=sample_rate
        if os.path.exists(self.save_audio_path):
            if not os.path.isdir(self.save_audio_path):
                raise ValueError("%f not a directory"%self.save_audio_path)
        else:
            os.makedirs(save_audio_path)
    def save(self,audio_ids, batch, adv_sig):
        bs = len(audio_ids)
        lengths = (batch.sig[0].size(1)*batch.sig[1]).long()
        for i in range(bs):
            id = audio_ids[i]
            wav = batch.sig[0][i,:lengths[i]]
            adv_wav = adv_sig[i,:lengths[i]]
            self.save_wav(id,wav,adv_wav)

    def save_wav(self,id,wav,adv_wav):
        nat_path = id + "_nat.wav"
        adv_path = id + "_adv.wav"
        torchaudio.save(nat_path,wav,self.sample_rate)
        torchaudio.save(adv_path,adv_wav,self.sample_rate)

