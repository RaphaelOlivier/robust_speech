"""
Multiple Brain classes that extend sb.Brain to enable attacks.
"""

import logging
import time
import warnings
import importlib
import os
from shutil import which
from xml.dom import NotFoundErr

import speechbrain as sb
import torch
from speechbrain.dataio.dataloader import LoopedLoader
from speechbrain.utils.distributed import run_on_main
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import robust_speech as rs
from robust_speech.adversarial.attacks.attacker import Attacker, TrainableAttacker
from robust_speech.adversarial.defenses.vote import ROVER_MAX_HYPS, ROVER_RECOMMENDED_HYPS, VoteEnsemble, Rover, MajorityVote
from robust_speech.adversarial.write_result import print_wer_summary, print_alignments, print_log_csv

import pdb
from copy import deepcopy

warnings.simplefilter("once", RuntimeWarning)

logger = logging.getLogger(__name__)

# Define write result procedure

class CustomErrorRateStats(sb.utils.metric_stats.ErrorRateStats):
    
    def write_stats(self, filestream, id = -1, batch = False):
        """Write all relevant info (e.g., error rate alignments) to file.
        * See MetricStats.write_stats()
        """
        
        # id = -1 for all audio in test set
        # batch = True write Log for each batch
        if not self.summary:
            self.summarize()
            
        if batch:
            print_log_csv(self.scores, id, filestream)
        else:
            print_wer_summary(self.summary, filestream)
            print_alignments(self.scores, filestream) 

# Define training procedure


class ASRBrain(sb.Brain):
    """
    Intermediate abstract brain class that specifies some methods for ASR models
     that can be attacked.
    See sb.Brain for more details.
    """

    def compute_forward(self, batch, stage):
        """Forward pass, to be overridden by sub-classes.

        Arguments
        ---------
        batch : torch.Tensor or tensors
            An element from the dataloader, including inputs for processing.
        stage : Union[sb.Stage, rs.Stage]
            The stage of the experiment:
            sb.Stage.TRAIN, sb.Stage.VALID, sb.Stage.TEST, rs.Stage.ATTACK

        Returns
        -------
        torch.Tensor or Tensors
            The outputs after all processing is complete.
            Directly passed to ``compute_objectives()``.
            In VALID or TEST stage, this should contain the predicted tokens.
            In ATTACK stage, batch.sig should be in the computation graph
            (no device change, no .detach())
        """
        raise NotImplementedError

    def compute_objectives(
        self, predictions, batch, stage, adv=False, targeted=False, reduction="mean"
    ):
        """Compute loss, to be overridden by sub-classes.

        Arguments
        ---------
        predictions : torch.Tensor or Tensors
            The output tensor or tensors to evaluate.
            Comes directly from ``compute_forward()``.
        batch : torch.Tensor or tensors
            An element from the dataloader, including targets for comparison.
        stage : Union[sb.Stage, rs.Stage]
            The stage of the experiment:
            sb.Stage.TRAIN, sb.Stage.VALID, sb.Stage.TEST, rs.Stage.ATTACK
        adv : bool
            Whether this is an adversarial input (used for metric logging)
        reduction : str
            the type of loss reduction to apply (required by some attacks)
        targeted : bool
            whether the attack is targeted

        Returns
        -------
        loss : torch.Tensor
            A tensor with the computed loss.
        """
        raise NotImplementedError

    def module_train(self):
        """
        Set PyTorch modules to training mode
        """
        self.modules.train()

    def module_eval(self):
        """
        Set PyTorch modules to eval mode
        """
        self.modules.eval()

    def get_tokens(self, predictions):
        """
        Extract tokens from predictions
        """
        return predictions[-1]


class PredictionEnsemble:
    """
    Iterable of predictions returned by EnsembleASRBrain
    """

    def __init__(self, predictions, ensemble_brain):
        self.predictions = predictions
        self.ensemble_brain = ensemble_brain

    def __getitem__(self, i):
        return self.predictions[i]

    def __len__(self):
        return len(self.predictions)


class AdvASRBrain(ASRBrain):
    """
    Intermediate abstract class that specifies some methods for ASR models
    that can be evaluated on attacks or trained adversarially.
    See sb.Brain for more details.

    Arguments
    ---------
    modules : dict of str:torch.nn.Module pairs
        These modules are passed to the optimizer by default if they have
        trainable parameters, and will have ``train()``/``eval()`` called on them.
    opt_class : torch.optim class
        A torch optimizer constructor that has takes only the list of
        parameters (e.g. a lambda or partial function definition). By default,
        this will be passed all modules in ``modules`` at the
        beginning of the ``fit()`` method. This behavior can be changed
        by overriding the ``configure_optimizers()`` method.
    hparams : dict
        Each key:value pair should consist of a string key and a hyperparameter
        that is used within the overridden methods. These will
        be accessible via an ``hparams`` attribute, using "dot" notation:
        e.g., self.hparams.model(x).
    run_opts : dict
        A set of options to change the runtime environment, including

        debug (bool)
            If ``True``, this will only iterate a few batches for all
            datasets, to ensure code runs without crashing.
        debug_batches (int)
            Number of batches to run in debug mode, Default ``2``.
        debug_epochs (int)
            Number of epochs to run in debug mode, Default ``2``.
            If a non-positive number is passed, all epochs are run.
        jit_module_keys (list of str)
            List of keys in ``modules`` that should be jit compiled.
        distributed_backend (str)
            One of ``nccl``, ``gloo``, ``mpi``.
        device (str)
            The location for performing computations.
        auto_mix_prec (bool)
            If ``True``, automatic mixed-precision is used.
            Activate it only with cuda.
        max_grad_norm (float)
            Default implementation of ``fit_batch()`` uses
            ``clip_grad_norm_`` with this value. Default: ``5``.
        nonfinite_patience (int)
            Number of times to ignore non-finite losses before stopping.
            Default: ``3``.
        noprogressbar (bool)
            Whether to turn off progressbar when training. Default: ``False``.
        ckpt_interval_minutes (float)
            Amount of time between saving intra-epoch checkpoints,
            in minutes, default: ``15.0``. If non-positive, these are not saved.

        Typically in a script this comes from ``speechbrain.parse_args``, which
        has different defaults than Brain. If an option is not defined here
        (keep in mind that parse_args will inject some options by default),
        then the option is also searched for in hparams (by key).
    checkpointer : speechbrain.Checkpointer
        By default, this will be used to load checkpoints, and will have the
        optimizer added to continue training if interrupted.
    attacker : Optional[robust_speech.adversarial.attacker.Attacker]
        If not None, this will run attacks on the nested source brain model
        (which may share its modules with this brain model)"""

    def __init__(  # noqa: C901
        self,
        modules=None,
        opt_class=None,
        hparams=None,
        run_opts=None,
        checkpointer=None,
        attacker=None,
    ):
        ASRBrain.__init__(
            self,
            modules=modules,
            opt_class=opt_class,
            hparams=hparams,
            run_opts=run_opts,
            checkpointer=checkpointer,
        )

        self.init_attacker(
            modules=modules,
            opt_class=opt_class,
            hparams=hparams,
            run_opts=run_opts,
            attacker=attacker,
        )
        self.voting_module = None
        if 'voting_module' in hparams:
            self.init_voting(hparams)
        self.tokenizer = None

    def __setattr__(self, name, value, attacker_brain=True):
        """Maintain similar attributes for the main and nested brain"""
        if (
            hasattr(self, "attacker")
            and self.attacker is not None
            and name != "attacker"
            and attacker_brain
        ):
            super(AdvASRBrain, self.attacker.asr_brain).__setattr__(name, value)
        super(AdvASRBrain, self).__setattr__(name, value)

    def init_voting(self, hparams):
        if "voting_iters" in hparams:
            if hparams["voting_iters"] <= 1:
                return
            self.voting_iters = hparams["voting_iters"]
        else:
            self.voting_iters = 3
        if 'rover_path' in hparams:
            rover_path = hparams['rover_path']
        elif os.environ.get('ROVER_PATH') is not None:
            rover_path = os.environ.get('ROVER_PATH')
        elif which('rover') is not None:
            rover_path = which('rover')
        else:
            return NotFoundErr('ROVER could not be found. Please follow instructions in README.md')
        self.voting_module = hparams['voting_module'](exec_path=rover_path)

    def init_attacker(
        self, modules=None, opt_class=None, hparams=None, run_opts=None, attacker=None
    ):
        """
        Initialize attacker class.
        Attackers take a brain as argument. If the attacker is not already instantiated,
         then it will receive a copy of the current object (without an attacker!),
         sharing modules. If the attacker is already instanciated
          it may contain a different brain. This is useful for
         transferring adversarial attacks between models:
         the noise is computed on the nested (source)
         brain and evaluated on the main (target) brain.
        """
        if isinstance(attacker, Attacker):  # attacker object already initiated
            self.attacker = attacker
        elif attacker is not None:  # attacker class
            brain_to_attack = type(self)(
                modules=modules,
                opt_class=opt_class,
                hparams=hparams,
                run_opts=run_opts,
                checkpointer=None,
                attacker=None,
            )
            self.attacker = attacker(brain_to_attack)
        else:
            self.attacker = None

    def compute_forward_adversarial(self, batch, stage):
        """Forward pass applied to an adversarial example.

        The default implementation depends on a few methods being defined
        with a particular behavior:

        * ``compute_forward()``

        Arguments
        ---------
        batch : torch.Tensor or tensors
            An element from the dataloader, including inputs for processing.
        stage : Stage
            The stage of the experiment: Stage.TRAIN, Stage.VALID, Stage.TEST

        Returns
        -------
        torch.Tensor or Tensors
            The outputs after all processing is complete.
            Directly passed to ``compute_objectives()``.
        """

        # assert stage != rs.Stage.ATTACK
        wavs = batch.sig[0]
        if self.attacker is not None:
            if stage == sb.Stage.TEST:
                adv_wavs = self.attacker.perturb_and_log(batch)
            else:
                adv_wavs = self.attacker.perturb(batch)
            adv_wavs = adv_wavs.detach()
            batch.sig = adv_wavs, batch.sig[1]
        res = self.compute_forward(batch, stage)
        batch.sig = wavs, batch.sig[1]
        return res, adv_wavs

    def fit_batch(self, batch):
        """Fit one batch, override to do multiple updates.

        The default implementation depends on a few methods being defined
        with a particular behavior:

        * ``compute_forward()``
        * ``compute_objectives()``

        Also depends on having optimizers passed at initialization.

        Arguments
        ---------
        batch : list of torch.Tensors
            Batch of data to use for training. Default implementation assumes
            this batch has two elements: inputs and targets.

        Returns
        -------
        detached loss
        """
        # Managing automatic mixed precision
        if self.auto_mix_prec:
            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = self.compute_forward(batch, sb.Stage.TRAIN)
                loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            if self.check_gradients(loss):
                self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            outputs = self.compute_forward(batch, sb.Stage.TRAIN)
            loss = self.compute_objectives(
                outputs, batch, sb.Stage.TRAIN, adv=False)

            # normalize the loss by gradient_accumulation step
            (loss / self.hparams.gradient_accumulation).backward()
            if self.step % self.hparams.gradient_accumulation == 0:
                # gradient clipping & early stop if loss is not fini
                self.check_gradients(loss)
                self.optimizer.step()
                self.optimizer.zero_grad()

            if self.check_gradients(loss):
                self.optimizer.step()
            self.optimizer.zero_grad()

        return loss.detach().cpu()

    def fit_batch_adversarial(self, batch):
        """Fit one batch with an adversarial objective,
        override to do multiple updates.

        The default implementation depends on a few methods being defined
        with a particular behavior:

        * ``compute_forward()``
        * ``compute_objectives()``

        Also depends on having optimizers passed at initialization.

        This method is currently under testing.

        Arguments
        ---------
        batch : list of torch.Tensors
            Batch of data to use for training. Default implementation assumes
            this batch has two elements: inputs and targets.

        Returns
        -------
        detached loss
        """
        warnings.warn(
            "Adversarial training is currently under development. \
            Use this function at your own discretion.",
            RuntimeWarning,
        )
        # Managing automatic mixed precision
        if self.auto_mix_prec:
            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs, _ = self.compute_forward_adversarial(
                    batch, sb.Stage.TRAIN)
                loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            if self.check_gradients(loss):
                self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            outputs, _ = self.compute_forward_adversarial(
                batch, sb.Stage.TRAIN)
            loss = self.compute_objectives(
                outputs, batch, sb.Stage.TRAIN, adv=True)

            # normalize the loss by gradient_accumulation step
            (loss / self.hparams.gradient_accumulation).backward()
            if self.step % self.hparams.gradient_accumulation == 0:
                # gradient clipping & early stop if loss is not fini
                self.check_gradients(loss)
                self.optimizer.step()
                self.optimizer.zero_grad()

            if self.check_gradients(loss):
                self.optimizer.step()
            self.optimizer.zero_grad()

        return loss.detach().cpu()

    def evaluate_batch(self, batch, stage):
        """Evaluate one batch, override for different procedure than train.

        The default implementation depends on two methods being defined
        with a particular behavior:

        * ``compute_forward()``
        * ``compute_objectives()``

        Arguments
        ---------
        batch : list of torch.Tensors
            Batch of data to use for evaluation. Default implementation assumes
            this batch has two elements: inputs and targets.
        stage : Stage
            The stage of the experiment: Stage.VALID, Stage.TEST

        Returns
        -------
        detached loss
        """
        if self.voting_module is not None:
            out = self.compute_forward_with_voting(
                batch, stage)
        else:
            out = self.compute_forward(batch, stage=stage)
        loss = self.compute_objectives(out, batch, stage=stage)
        return loss.detach().cpu()

    def evaluate_batch_adversarial(self, batch, stage, target=None):
        """Evaluate one batch on adversarial examples.

        The default implementation depends on two methods being defined
        with a particular behavior:

        * ``compute_forward()``
        * ``compute_objectives()``

        Arguments
        ---------
        batch : list of torch.Tensors
            Batch of data to use for evaluation. Default implementation assumes
            this batch has two elements: inputs and targets.
        stage : Stage
            The stage of the experiment: Stage.VALID, Stage.TEST
        target : str
            The optional attack target

        Returns
        -------
        detached loss
        """
        tokenizer = (
            self.tokenizer if hasattr(
                self, "tokenizer") else self.hparams.tokenizer
        )
        if target is not None and self.attacker.targeted:
            batch_to_attack = target.replace_tokens_in_batch(
                batch, tokenizer, self.hparams
            )
        else:
            batch_to_attack = batch
        #print("original input",adv_wav.mean())
        predictions, adv_wav = self.compute_forward_adversarial(
            batch_to_attack, stage=stage
        )
        #print("generated or loaded adv",adv_wav.mean())
        advloss, targetloss = None, None
        with torch.no_grad():
            targeted = target is not None and self.attacker.targeted
            batch_to_attack.sig = adv_wav, batch_to_attack.sig[1]

            if self.voting_module is not None:
                predictions = self.compute_forward_with_voting(
                    batch_to_attack, stage)

            loss = self.compute_objectives(
                predictions, batch_to_attack, stage=stage, adv=True, targeted=targeted
            ).detach()
            if targeted:
                targetloss = loss
                batch.sig = adv_wav, batch.sig[1]
                if self.voting_module is not None:
                    predictions = self.compute_forward_with_voting(
                        batch, stage)
                else:
                    predictions = self.compute_forward(batch, stage=stage)
                advloss = self.compute_objectives(
                    predictions, batch, stage=stage, adv=True, targeted=False
                ).detach()
                batch.sig = batch_to_attack.sig
            else:
                advloss = loss

        return advloss, targetloss

    def compute_forward_with_voting(self, batch, stage):
        preds = []
        for i in range(self.voting_iters):
            predictions = self.compute_forward(
                batch, stage=stage)
            predicted_tokens = predictions[-1][0]
            predicted_tokens = [str(s) for s in predicted_tokens]
            predicted_words = " ".join(predicted_tokens)
            preds.append([predicted_words])
        outs = self.voting_module.run(preds)
        outs = outs[0].split(" ")
        tokens = [int(token) for token in outs]

        predictions = list(predictions)
        predictions[-1] = [tokens]
        predictions = tuple(predictions)

        return predictions

    def fit(
        self,
        epoch_counter,
        train_set,
        valid_set=None,
        progressbar=None,
        train_loader_kwargs={},
        valid_loader_kwargs={},
    ):
        """Iterate epochs and datasets to improve objective.

        Relies on the existence of multiple functions that can (or should) be
        overridden. The following methods are used and expected to have a
        certain behavior:

        * ``fit_batch()``
        * ``evaluate_batch()``
        * ``fit_batch_adversarial()``
        * ``evaluate_batch_adversarial()``
        * ``update_average()``

        If the initialization was done with distributed_count > 0 and the
        distributed_backend is ddp, this will generally handle multiprocess
        logic, like splitting the training data into subsets for each device and
        only saving a checkpoint on the main process.

        Arguments
        ---------
        epoch_counter : iterable
            Each call should return an integer indicating the epoch count.
        train_set : Dataset, DataLoader
            A set of data to use for training. If a Dataset is given, a
            DataLoader is automatically created. If a DataLoader is given, it is
            used directly.
        valid_set : Dataset, DataLoader
            A set of data to use for validation. If a Dataset is given, a
            DataLoader is automatically created. If a DataLoader is given, it is
            used directly.
        train_loader_kwargs : dict
            Kwargs passed to `make_dataloader()` for making the train_loader
            (if train_set is a Dataset, not DataLoader).
            E.G. batch_size, num_workers.
            DataLoader kwargs are all valid.
        valid_loader_kwargs : dict
            Kwargs passed to `make_dataloader()` for making the valid_loader
            (if valid_set is a Dataset, not DataLoader).
            E.g., batch_size, num_workers.
            DataLoader kwargs are all valid.
        progressbar : bool
            Whether to display the progress of each epoch in a progressbar.
        """

        if not (
            isinstance(train_set, DataLoader) or isinstance(
                train_set, LoopedLoader)
        ):
            train_set = self.make_dataloader(
                train_set, stage=sb.Stage.TRAIN, **train_loader_kwargs
            )
        if valid_set is not None and not (
            isinstance(valid_set, DataLoader) or isinstance(
                valid_set, LoopedLoader)
        ):
            valid_set = self.make_dataloader(
                valid_set,
                stage=sb.Stage.VALID,
                ckpt_prefix=None,
                **valid_loader_kwargs,
            )

        self.on_fit_start()

        if progressbar is None:
            progressbar = not self.noprogressbar

        # Iterate epochs
        for epoch in epoch_counter:
            # Training stage
            self.on_stage_start(sb.Stage.TRAIN, epoch)
            self.modules.train()

            # Reset nonfinite count to 0 each epoch
            self.nonfinite_count = 0

            if self.train_sampler is not None and hasattr(
                self.train_sampler, "set_epoch"
            ):
                self.train_sampler.set_epoch(epoch)

            # Time since last intra-epoch checkpoint
            last_ckpt_time = time.time()

            # Only show progressbar if requested and main_process
            enable = progressbar and sb.utils.distributed.if_main_process()
            with tqdm(
                train_set,
                initial=self.step,
                dynamic_ncols=True,
                disable=not enable,
            ) as pbar:
                for batch in pbar:
                    self.step += 1
                    if self.attacker is not None:
                        loss = self.fit_batch_adversarial(batch)
                    else:
                        loss = self.fit_batch(batch)
                    self.avg_train_loss = self.update_average(
                        loss, self.avg_train_loss)
                    if self.attacker is not None:
                        pbar.set_postfix(adv_train_loss=self.avg_train_loss)
                    else:
                        pbar.set_postfix(train_loss=self.avg_train_loss)

                    # Debug mode only runs a few batches
                    if self.debug and self.step == self.debug_batches:
                        break

                    if (
                        self.checkpointer is not None
                        and self.ckpt_interval_minutes > 0
                        and time.time() - last_ckpt_time
                        >= self.ckpt_interval_minutes * 60.0
                    ):
                        # This should not use run_on_main, because that
                        # includes a DDP barrier. That eventually leads to a
                        # crash when the processes'
                        # time.time() - last_ckpt_time differ and some
                        # processes enter this block while others don't,
                        # missing the barrier.
                        if sb.utils.distributed.if_main_process():
                            self._save_intra_epoch_ckpt()
                        last_ckpt_time = time.time()

            # Run train "on_stage_end" on all processes
            self.on_stage_end(sb.Stage.TRAIN, self.avg_train_loss, epoch)
            self.avg_train_loss = 0.0
            self.step = 0

            # Validation stage
            if valid_set is not None:
                self.on_stage_start(sb.Stage.VALID, epoch)
                self.modules.eval()
                avg_valid_loss = 0.0
                avg_valid_adv_loss = None
                if self.attacker is not None:
                    avg_valid_adv_loss = 0.0
                for batch in tqdm(valid_set, dynamic_ncols=True, disable=not enable):
                    self.step += 1
                    loss = self.evaluate_batch(batch, stage=sb.Stage.VALID)
                    avg_valid_loss = self.update_average(loss, avg_valid_loss)
                    if self.attacker is not None:
                        adv_loss, _ = self.evaluate_batch_adversarial(
                            batch, stage=sb.Stage.VALID
                        )
                        avg_valid_adv_loss = self.update_average(
                            adv_loss, avg_valid_adv_loss
                        )

                    # Debug mode only runs a few batches
                    if self.debug and self.step == self.debug_batches:
                        break

                # Only run validation "on_stage_end" on main process
                self.step = 0
                run_on_main(
                    self.on_stage_end,
                    args=[sb.Stage.VALID, avg_valid_loss, epoch],
                    kwargs={"stage_adv_loss": avg_valid_adv_loss},
                )

            # Debug mode only runs a few epochs
            if self.debug and epoch == self.debug_epochs:
                break

    def evaluate(
        self,
        test_set,
        max_key=None,
        min_key=None,
        progressbar=None,
        test_loader_kwargs={},
        save_audio_path=None,
        load_audio=False,
        sample_rate=16000,
        target=None,
    ):
        """Iterate test_set and evaluate brain performance. By default, loads
        the best-performing checkpoint (as recorded using the checkpointer).
        Arguments
        ---------
        test_set : Dataset, DataLoader
            If a DataLoader is given, it is iterated directly. Otherwise passed
            to ``self.make_dataloader()``.
        max_key : str
            Key to use for finding best checkpoint, passed to
            ``on_evaluate_start()``.
        min_key : str
            Key to use for finding best checkpoint, passed to
            ``on_evaluate_start()``.
        progressbar : bool
            Whether to display the progress in a progressbar.
        test_loader_kwargs : dict
            Kwargs passed to ``make_dataloader()`` if ``test_set`` is not a
            DataLoader. NOTE: ``loader_kwargs["ckpt_prefix"]`` gets
            automatically overwritten to ``None`` (so that the test DataLoader
            is not added to the checkpointer).
        save_audio_path : str
            optional path where to store adversarial audio files
        load_audio : bool
            whether to load audio files from save_audio_path instead of running the attack
        sample_rate = 16000
            the audio sample rate
        target : str
            The optional attack target
        Returns
        -------
        average test loss
        """
        if progressbar is None:
            progressbar = not self.noprogressbar

        if not (isinstance(test_set, DataLoader) or isinstance(test_set, LoopedLoader)):
            test_loader_kwargs["ckpt_prefix"] = None
            test_set = self.make_dataloader(
                test_set, sb.Stage.TEST, **test_loader_kwargs
            )
            
        self.on_evaluate_start(max_key=max_key, min_key=min_key)
        self.on_stage_start(sb.Stage.TEST, epoch=None)
        self.modules.eval()
        avg_test_loss = 0.0
        avg_test_adv_loss = None
        avg_test_adv_loss_target = None
        if self.attacker is not None:
            avg_test_adv_loss = 0.0
            self.attacker.on_evaluation_start(
                load_audio=load_audio, save_audio_path=save_audio_path)
            
        for batch in tqdm(test_set, dynamic_ncols=True, disable=not progressbar):
            self.step += 1
            # Calculate the loss after attacking (Ground truth and Predict)
            loss = self.evaluate_batch(batch, stage=sb.Stage.TEST)
            avg_test_loss = self.update_average(loss, avg_test_loss)

            if self.attacker is not None:
                # Loop through the number of iterations
                adv_loss, adv_loss_target = self.evaluate_batch_adversarial(
                    batch, stage=sb.Stage.TEST, target=target
                )
                avg_test_adv_loss = self.update_average(
                    adv_loss, avg_test_adv_loss)
                if adv_loss_target:
                    if avg_test_adv_loss_target is None:
                        avg_test_adv_loss_target = 0.0
                    avg_test_adv_loss_target = self.update_average(
                        adv_loss_target, avg_test_adv_loss_target
                    )

            # Debug mode only runs a few batches
            if self.debug and self.step == self.debug_batches:
                break

            # Only run evaluation "on_stage_end" on main process
            run_on_main(
                self.on_batch_end,
                args=[sb.Stage.TEST, avg_test_loss, batch.id[0], None],
                kwargs={
                    "stage_adv_loss": avg_test_adv_loss,
                    "stage_adv_loss_target": avg_test_adv_loss_target,
                },
            )
        run_on_main(
            self.on_stage_end,
            args=[sb.Stage.TEST, avg_test_loss, None],
            kwargs={
                "stage_adv_loss": avg_test_adv_loss,
                "stage_adv_loss_target": avg_test_adv_loss_target,
            },
        )
        self.step = 0
        self.on_evaluate_end()
        return avg_test_loss

    def fit_attacker(
        self,
        fit_set,
        max_key=None,
        min_key=None,
        progressbar=None,
        loader_kwargs={},
    ):
        if progressbar is None:
            progressbar = not self.noprogressbar

        if not (isinstance(fit_set, DataLoader) or isinstance(fit_set, LoopedLoader)):
            loader_kwargs["ckpt_prefix"] = None
            fit_set = self.make_dataloader(
                fit_set, sb.Stage.TEST, **loader_kwargs
            )
        self.on_evaluate_start(max_key=max_key, min_key=min_key)
        self.on_stage_start(sb.Stage.TEST, epoch=None)
        self.modules.eval()
        avg_test_loss = 0.0

        if self.attacker is None:
            raise ValueError("No attacker to train!")
        if not isinstance(self.attacker, TrainableAttacker):
            raise ValueError("fit_attacker cannot be called for non-trainable attack %s" %
                             self.attacker.__class__.__name__)

        self.attacker.on_fit_start()

        self.attacker.fit(fit_set)

        self.attacker.on_fit_end()

        return avg_test_loss

    def on_stage_start(self, stage, epoch):
        """Gets called when a stage starts.

        Useful for defining class variables used during the stage.

        Arguments
        ---------
        stage : Stage
            The stage of the experiment: Stage.TRAIN, Stage.VALID, Stage.TEST
        epoch : int
            The current epoch count.
        """
        if stage != sb.Stage.TRAIN:
            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = CustomErrorRateStats()
            # self.wer_metric = self.hparams.error_rate_computer()
            self.adv_cer_metric = self.hparams.cer_computer()
            self.adv_wer_metric = CustomErrorRateStats()
            # self.adv_wer_metric = self.hparams.error_rate_computer()
            self.adv_cer_metric_target = self.hparams.cer_computer()
            self.adv_wer_metric_target = CustomErrorRateStats()
            # self.adv_wer_metric_target = self.hparams.error_rate_computer()
            try:
                self.adv_ser_metric_target = self.hparams.ser_computer()
            except:
                self.adv_ser_metric_target = None

    def on_batch_end(self, state, stage_loss, id, epoch, stage_adv_loss=None, stage_adv_loss_target=None):
        # log_csv = self.hparams.wer_file.split('.')[0] + '.csv'
        log_csv = os.path.splitext(self.hparams.wer_file)[0] + '.wer.csv'
        with open(log_csv, "a+") as log:
            self.wer_metric.write_stats(log, id=id, batch=True)
            # self.adv_wer_metric.write_stats(log, id=id, batch=True)
            # self.adv_wer_metric_target.write_stats(log, id=id, batch=True)
        self.attacker.on_batch_end(self.hparams)
        
    def on_stage_end(
        self, stage, stage_loss, epoch, stage_adv_loss=None, stage_adv_loss_target=None
    ):
        """Gets called at the end of a stage.

        Useful for computing stage statistics, saving checkpoints, etc.

        Arguments
        ---------
        stage : Stage
            The stage of the experiment: Stage.TRAIN, Stage.VALID, Stage.TEST
        stage_loss : float
            The average loss over the completed stage.
        epoch : int
            The current epoch count.
        stage_adv_loss : Optional[float]
            The average adversarial loss over the completed stage, if available.
        """
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage_adv_loss is not None:
            stage_stats["adv loss"] = stage_adv_loss
        if stage_adv_loss_target is not None:
            stage_stats["adv loss target"] = stage_adv_loss_target
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")
            if stage_adv_loss is not None:
                stage_stats["adv CER"] = self.adv_cer_metric.summarize(
                    "error_rate")
                stage_stats["adv WER"] = self.adv_wer_metric.summarize(
                    "error_rate")
            if stage_adv_loss_target is not None:
                stage_stats["adv CER target"] = self.adv_cer_metric_target.summarize(
                    "error_rate"
                )
                stage_stats["adv WER target"] = self.adv_wer_metric_target.summarize(
                    "error_rate"
                )
                if self.adv_ser_metric_target is not None:
                    stage_stats["adv SER target"] = self.adv_ser_metric_target.summarize(
                        "error_rate"
                    )

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            if hasattr(self.hparams, "lr_annealing"):
                old_lr, new_lr = self.hparams.lr_annealing(stage_stats["WER"])
                sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
            else:
                old_lr = self.optimizer.param_groups[0]["lr"]
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"WER": stage_stats["WER"]},
                min_keys=["WER"],
                num_to_keep=self.hparams.num_to_keep if "num_to_keep" in self.hparams.__dict__ else 1
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Evaluation stage": "TEST"},
                test_stats=stage_stats,
            )
            with open(self.hparams.wer_file, "w") as wer:
                self.wer_metric.write_stats(wer)
                # self.adv_wer_metric.write_stats(wer)
                # self.adv_wer_metric_target.write_stats(wer)

    def on_evaluate_start(self, max_key=None, min_key=None):
        """Run at the beginning of evlauation.
        Sets attack metrics and loggers"""
        super().on_evaluate_start(max_key=max_key, min_key=min_key)
        if self.attacker is not None:
            self.attacker.on_evaluation_start()

    def on_evaluate_end(self):
        """Run at the beginning of evlauation.
        Log attack metrics and save perturbed audio"""
        if self.attacker is not None:
            self.attacker.on_evaluation_end(self.hparams.train_logger)

    def compute_forward(self, batch, stage):
        """Forward pass, to be overridden by sub-classes.

        Arguments
        ---------
        batch : torch.Tensor or tensors
            An element from the dataloader, including inputs for processing.
        stage : Union[sb.Stage, rs.Stage]
            The stage of the experiment:
            sb.Stage.TRAIN, sb.Stage.VALID, sb.Stage.TEST, rs.Stage.ATTACK

        Returns
        -------
        torch.Tensor or Tensors
            The outputs after all processing is complete.
            Directly passed to ``compute_objectives()``.
            In VALID or TEST stage, this should contain the predicted tokens.
            In ATTACK stage, batch.sig should be in the computation graph
            (no device change, no .detach())
        """
        raise NotImplementedError

    def compute_objectives(
        self, predictions, batch, stage, adv=False, targeted=False, reduction="mean"
    ):
        """Compute loss, to be overridden by sub-classes.

        Arguments
        ---------
        predictions : torch.Tensor or Tensors
            The output tensor or tensors to evaluate.
            Comes directly from ``compute_forward()``.
        batch : torch.Tensor or tensors
            An element from the dataloader, including targets for comparison.
        stage : Union[sb.Stage, rs.Stage]
            The stage of the experiment:
            sb.Stage.TRAIN, sb.Stage.VALID, sb.Stage.TEST, rs.Stage.ATTACK
        adv : bool
            Whether this is an adversarial input (used for metric logging)
        reduction : str
            the type of loss reduction to apply (required by some attacks)
        targeted : bool
            whether the attack is targeted

        Returns
        -------
        loss : torch.Tensor
            A tensor with the computed loss.
        """
        raise NotImplementedError


class EnsembleASRBrain(AdvASRBrain):
    """
    Ensemble of multiple brains.
    This class is used for attacks that compute adversarial noise
    simultaneously on multiple models.
    """

    def __init__(self, asr_brains, ref_valid_test=-1, ref_attack=None, ref_train=None):
        self.asr_brains = asr_brains
        self.ref_valid_test = ref_valid_test  # use this model to return tokens
        self.ref_attack = ref_attack
        self.ref_train = ref_train

    @property
    def nmodels(self):
        """Number of models in the ensemble"""
        return len(self.asr_brains)

    def compute_forward(self, batch, stage, model_idx=None):
        """
        forward pass of all  or one model(s)
        """
        # concatenate predictions
        if model_idx is not None:
            return self.asr_brains[model_idx].compute_forward(batch, stage)
        elif stage == rs.Stage.ATTACK and self.ref_attack is not None:
            return self.asr_brains[self.ref_attack].compute_forward(batch, stage)
        elif stage == sb.Stage.TRAIN and self.ref_train is not None:
            return self.asr_brains[self.ref_train].compute_forward(batch, stage)
        elif stage in [sb.Stage.VALID, sb.Stage.TEST] and self.ref_valid_test is not None:
            return self.asr_brains[self.ref_valid_test].compute_forward(batch, stage)
        predictions = []
        for asr_brain in self.asr_brains:
            pred = asr_brain.compute_forward(batch, stage)
            predictions.append(pred)
        predictions = PredictionEnsemble(predictions, ensemble_brain=self)
        return predictions

    def get_tokens(self, predictions, all_models=False, model_idx=None):
        """
        Extract tokens from predictions.

        :param predictions: model predictions
        :param all: whether to extract all tokens or just one
        :param model_idx: which model to extract tokens from
        (defaults to self.ref_train, self.ref_attack or self.valid_test depending on stage)
        """
        if isinstance(predictions, PredictionEnsemble) and predictions.ensemble_brain == self:
            assert len(predictions) == self.nmodels
            if all:
                return [
                    self.asr_brains[i].get_tokens(pred)
                    for i, pred in enumerate(predictions)
                ]
            if model_idx is not None:
                return self.asr_brains[model_idx].get_tokens(predictions[model_idx])
            return self.asr_brains[self.ref_valid_test].get_tokens(
                predictions[self.ref_valid_test]
            )
        return self.asr_brains[self.ref_valid_test].get_tokens(predictions)

    def compute_objectives(
        self,
        predictions,
        batch,
        stage,
        adv=False,
        targeted=False,
        reduction="mean",
        average=True,
        model_idx=None,
    ):
        """
        Compute the losses of all or one model
        """
        # concatenate of average objectives
        if (
            isinstance(
                predictions, PredictionEnsemble) and predictions.ensemble_brain == self
        ):  # many predictions
            assert len(predictions) == self.nmodels
            losses = []
            for i in range(self.nmodels):
                # one pred per model or n pred per model
                asr_brain = (
                    self.asr_brains[i]
                    if model_idx is None
                    else self.asr_brains[model_idx]
                )
                pred = (
                    predictions[i]
                    if isinstance(predictions, PredictionEnsemble)
                    else predictions
                )
                loss = asr_brain.compute_objectives(
                    pred, batch, stage, adv=adv, reduction=reduction
                )
                losses.append(loss)
            losses = torch.stack(losses, dim=0)
            if average:
                return torch.mean(losses, dim=0)
            return losses
        if model_idx is None:
            if stage == rs.Stage.ATTACK and self.ref_attack is not None:
                model_idx = self.ref_attack
            elif stage == sb.Stage.TRAIN and self.ref_train is not None:
                model_idx = self.ref_train
            elif (stage == sb.Stage.VALID or stage == sb.Stage.TEST) and self.ref_train is not None:
                model_idx = self.ref_valid_test
        return self.asr_brains[model_idx].compute_objectives(
            predictions, batch, stage, adv=adv, targeted=targeted, reduction=reduction
        )

    def __setattr__(self, name, value):  # useful to set tokenizer
        if name not in ["asr_brains", "ref_attack", "ref_train", "ref_valid_test"]:
            for brain in self.asr_brains:
                brain.__setattr__(name, value)
        super(EnsembleASRBrain, self).__setattr__(name, value)

    def module_train(self):
        """
        Set PyTorch modules to training mode
        """
        for brain in self.asr_brains:
            brain.module_train()

    def module_eval(self):
        """
        Set PyTorch modules to eval mode
        """
        for brain in self.asr_brains:
            brain.module_eval()

    @property
    def device(self):
        return self.asr_brains[0].device
