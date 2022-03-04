from advertorch.attacks.base import Attack,LabelMixin
from robust_speech.adversarial.metrics import SNRComputer
class Attacker(Attack,LabelMixin):
    def on_evaluation_start(self):
        self.snr_metric = SNRComputer()

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
        return adv_wav