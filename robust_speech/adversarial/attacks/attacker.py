from advertorch.attacks.base import Attack,LabelMixin
from robust_speech.adversarial.metrics import SNRComputer, AudioSaver
class Attacker(Attack,LabelMixin):
    def on_evaluation_start(self, save_audio_path=None,sample_rate=16000):
        self.snr_metric = SNRComputer()
        self.save_audio_path = save_audio_path
        if self.save_audio_path:
            self.audio_saver = AudioSaver(save_audio_path, sample_rate)

    def on_evaluation_end(self, logger):
        snr = self.snr_metric.summarize()
        snr = {"average":snr["average"], 'min_score':snr['min_score'], 'max_score':snr['max_score']}
        logger.log_stats(
                stats_meta={},
                test_stats={"Adversarial SNR":snr},
            )

    def perturb_and_log(self,batch):
        adv_wav = self.perturb(batch)
        self.snr_metric.append(batch.id, batch,adv_wav)
        if self.save_audio_path:
            self.audio_saver.save(batch.id,batch,adv_wav)
        return adv_wav