import time

import numpy
import torch
from torch.nn.utils import clip_grad_norm_

from modules.training.base_trainer import BaseTrainer
from utils._logging import epoch_progress
from utils.training import save_checkpoint


class AdversarialTrainer(BaseTrainer):
    """
    An abstract class representing a Trainer.
    A Trainer object, is responsible for handling the training process and
    provides various helper methods.

    All other trainers should subclass it.
    All subclasses should override process_batch, which handles the way
    you feed the input data to the model and performs a forward pass.
    """

    def __init__(self, model, adversary,
                 train_loader, valid_loader,
                 model_criterion, adversary_criterion,
                 model_optimizers, adversary_optimizers, config, device,
                 batch_end_callbacks=None, loss_weights=None, **kwargs):

        super().__init__(train_loader, valid_loader, config, device,
                         batch_end_callbacks, loss_weights, **kwargs)

        self.model = model
        self.adversary = adversary
        self.model_criterion = model_criterion
        self.adversary_criterion = adversary_criterion
        self.model_optimizers = model_optimizers
        self.adversary_optimizers = adversary_optimizers

        if not isinstance(self.model_optimizers, (tuple, list)):
            self.model_optimizers = [self.model_optimizers]

        if not isinstance(self.adversary_optimizers, (tuple, list)):
            self.adversary_optimizers = [self.adversary_optimizers]

    def _process_model_batch(self, *args):
        raise NotImplementedError

    def _process_adversary_batch(self, *args):
        raise NotImplementedError

    def _compute_model_loss(self, *args):
        raise NotImplementedError

    def _compute_adversary_loss(self, *args):
        raise NotImplementedError

    def _seq_loss(self, logits, labels):
        loss = self.model_criterion(
            logits.contiguous().view(-1, logits.size(-1)),
            labels.contiguous().view(-1))
        return loss

    def _zero_gradients(self):
        for optimizer in self.model_optimizers:
            optimizer.zero_grad()
        for optimizer in self.adversary_optimizers:
            optimizer.zero_grad()

    def train_epoch(self):
        """
        Train the network for one epoch and return the average loss.
        * This will be a pessimistic approximation of the true loss
        of the network, as the loss of the first batches will be higher
        than the true.

        Returns:
            loss (float, list(float)): list of mean losses

        """
        self.model.train()
        self.adversary.train()
        total_model_losses = []
        total_adv_losses = []

        self.epoch += 1
        epoch_start = time.time()

        if isinstance(self.train_loader, (tuple, list)):
            iterator = zip(*self.train_loader)
        else:
            iterator = self.train_loader

        for i_batch, batch in enumerate(iterator, 1):

            self.step += 1
            batch = self._batch_to_device(batch)

            # ------------------------------------------------------ #
            #   Train the model
            # ------------------------------------------------------ #

            # zero gradients
            self._zero_gradients()

            m_outs = self._process_model_batch(*batch)

            model_losses = self._compute_model_loss(m_outs, batch)

            model_loss_sum, model_losses = self._aggregate_losses(model_losses,
                                                                  [1, 1])

            # back-propagate
            model_loss_sum.backward()

            if self.clip is not None:
                for optimizer in self.model_optimizers:
                    clip_grad_norm_((p for group in optimizer.param_groups
                                     for p in group['params']), self.clip)

            # update weights
            for optimizer in self.model_optimizers:
                optimizer.step()

            # ------------------------------------------------------ #
            #   Train the discriminator
            # ------------------------------------------------------ #

            # zero gradients
            self._zero_gradients()

            adv_losses = self._compute_adversary_loss(m_outs, batch)
            adv_loss_sum, adv_losses = self._aggregate_losses(adv_losses)

            # back-propagate
            adv_loss_sum.backward()

            if self.clip is not None:
                for optimizer in self.adversary_optimizers:
                    clip_grad_norm_((p for group in optimizer.param_groups
                                     for p in group['params']), self.clip)

            # update weights
            for optimizer in self.adversary_optimizers:
                optimizer.step()

            # ------------------------------------------------------ #
            #   Logging
            # ------------------------------------------------------ #
            total_model_losses.append(model_losses)
            total_adv_losses.append(adv_losses)
            if self.step % self.log_interval == 0:
                self.progress_log = epoch_progress(self.epoch, i_batch,
                                                   self.batch_size,
                                                   self.train_set_size,
                                                   epoch_start)

            for c in self.batch_end_callbacks:
                if callable(c):
                    c(i_batch, total_model_losses, total_adv_losses, m_outs)

        return numpy.array(total_model_losses).mean(axis=0), \
               numpy.array(total_adv_losses).mean(axis=0)

    def eval_epoch(self):
        """
        Evaluate the network for one epoch and return the average loss.

        Returns:
            loss (float, list(float)): list of mean losses

        """
        self.model.eval()
        losses = []

        if isinstance(self.valid_loader, (tuple, list)):
            iterator = zip(*self.valid_loader)
        else:
            iterator = self.valid_loader

        with torch.no_grad():
            for i_batch, batch in enumerate(iterator, 1):

                # move all tensors in batch to the selected device
                if isinstance(self.valid_loader, (tuple, list)):
                    batch = list(map(lambda x:
                                     list(map(lambda y: y.to(self.device), x)),
                                     batch))
                else:
                    batch = list(map(lambda x: x.to(self.device), batch))

                    m_outs = self._process_model_batch(*batch)
                    _losses = self._compute_model_loss(m_outs, batch)
                    _loss_sum, _losses = self._aggregate_losses(_losses)
                    losses.append(_losses)

        return numpy.array(losses).mean(axis=0)

    def get_state(self):
        if self.train_loader.dataset.subword:
            _vocab = self.train_loader.dataset.subword_path

            _vocab = (self.config["vocab"]["src"]["subword_path"],
                      self.config["vocab"]["trg"]["subword_path"])
        else:
            _vocab = (self.train_loader.dataset.src_vocab,
                      self.train_loader.dataset.trg_vocab)

        state = {
            "config": self.config,
            "epoch": self.epoch,
            "step": self.step,
            "model": self.model.state_dict(),
            "model_class": self.model.__class__.__name__,
            "adversary": self.adversary.state_dict(),
            "adversary_class": self.adversary.__class__.__name__,
            "model_optimizers": [x.state_dict()
                                 for x in self.model_optimizers],
            "adversary_optimizers": [x.state_dict()
                                     for x in self.adversary_optimizers],
            "vocab": _vocab,
        }

        return state

    def checkpoint(self, name=None, timestamp=False, tags=None, verbose=False):

        if name is None:
            name = self.config["name"]

        return save_checkpoint(self.get_state(),
                               name=name, tag=tags, timestamp=timestamp,
                               verbose=verbose)