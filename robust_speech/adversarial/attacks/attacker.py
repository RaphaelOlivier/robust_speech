from advertorch.attacks.base import Attack,LabelMixin
from robust_speech.adversarial.metrics import SNRComputer, AudioSaver
class Attacker(Attack,LabelMixin):
    """
    Abstract class for running attacks and logging results (SNR and audio files)

    Arguments
    ---------
    asr_brain: rs.adversarial.brain.ASRBrain
        brain object.
    targeted: bool
        if the attack is targeted.
    """

    def on_evaluation_start(self, save_audio_path=None,sample_rate=16000):
        """
        Method to run at the beginning of an evaluation phase with adverersarial attacks.

        Arguments
        ---------
        save_audio_path: optional string
            path to the folder in which to save audio files
        sample_rate: int
            audio sample rate for wav encoding
        """
        self.snr_metric = SNRComputer()
        self.save_audio_path = save_audio_path
        if self.save_audio_path:
            self.audio_saver = AudioSaver(save_audio_path, sample_rate)

    def on_evaluation_end(self, logger):
        """
        Method to run at the end of an evaluation phase with adverersarial attacks.

        Arguments
        ---------
        logger: sb.utils,train_logger.FileLogger
            path to the folder in which to save audio files
        """
        snr = self.snr_metric.summarize()
        snr = {"average":snr["average"], 'min_score':snr['min_score'], 'max_score':snr['max_score']}
        logger.log_stats(
                stats_meta={},
                test_stats={"Adversarial SNR":snr},
            )

    def perturb_and_log(self,batch):
        """
        Compute an adversarial perturbation and log results

        Arguments
        ---------
        batch : sb.PaddedBatch
            The input batch to perturb

        Returns
        -------
        the tensor of the perturbed batch
        """
        adv_wav = self.perturb(batch)
        self.snr_metric.append(batch.id, batch,adv_wav)
        if self.save_audio_path:
            self.audio_saver.save(batch.id,batch,adv_wav)
        return adv_wav

    def perturb(self, batch):
        """
        Compute an adversarial perturbation

        Arguments
        ---------
        batch : sb.PaddedBatch
            The input batch to perturb

        Returns
        -------
        the tensor of the perturbed batch
        """

        raise NotImplementedError