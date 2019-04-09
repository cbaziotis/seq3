import time

import numpy
import torch
from torch.nn.utils import clip_grad_norm_

from modules.training.base_trainer import BaseTrainer
from utils._logging import epoch_progress
from utils.training import save_checkpoint


class Trainer(BaseTrainer):
    """
    An abstract class representing a Trainer.
    A Trainer object, is responsible for handling the training process and
    provides various helper methods.

    All other trainers should subclass it.
    All subclasses should override process_batch, which handles the way
    you feed the input data to the model and performs a forward pass.
    """

    def __init__(self, model, train_loader, valid_loader, criterion,
                 optimizers, config, device,
                 batch_end_callbacks=None, loss_weights=None, **kwargs):

        super().__init__(train_loader, valid_loader, config, device,
                         batch_end_callbacks, loss_weights, **kwargs)

        self.model = model
        self.criterion = criterion
        self.optimizers = optimizers

        if not isinstance(self.optimizers, (tuple, list)):
            self.optimizers = [self.optimizers]

    def _process_batch(self, *args):
        raise NotImplementedError

    def _seq_loss(self, logits, labels):

        """
        Compute a sequence loss (i.e. per timestep).
        Used for tasks such as Translation, Language Modeling and
        Sequence Labelling.
        """
        _logits = logits.contiguous().view(-1, logits.size(-1))
        _labels = labels.contiguous().view(-1)
        loss = self.criterion(_logits, _labels)

        return loss

    def grads(self):
        """
        Get the list of the norms of the gradients for each parameter
        """
        return [(name, parameter.grad.norm().item())
                for name, parameter in self.model.named_parameters()
                if parameter.requires_grad and parameter.grad is not None]

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
        losses = []

        self.epoch += 1
        epoch_start = time.time()

        iterator = self._dataset_iterator(self.train_loader)
        for i_batch, batch in enumerate(iterator, 1):

            self.step += 1

            # zero gradients
            for optimizer in self.optimizers:
                optimizer.zero_grad()

            batch = self._batch_to_device(batch)

            # return here only the first batch losses, in order to avoid
            # breaking the existing framework
            batch_losses, batch_outputs = self._process_batch(*batch)

            # aggregate the losses into a single loss value
            loss_sum, loss_list = self._aggregate_losses(batch_losses)
            losses.append(loss_list)

            # back-propagate
            loss_sum.backward()

            if self.clip is not None:
                # clip_grad_norm_(self.model.parameters(), self.clip)
                for optimizer in self.optimizers:
                    clip_grad_norm_((p for group in optimizer.param_groups
                                     for p in group['params']), self.clip)

            # update weights
            for optimizer in self.optimizers:
                optimizer.step()

            if self.step % self.log_interval == 0:
                self.progress_log = epoch_progress(self.epoch, i_batch,
                                                   self.batch_size,
                                                   self.train_set_size,
                                                   epoch_start)

            for c in self.batch_end_callbacks:
                if callable(c):
                    c(batch, losses, loss_list, batch_outputs)
        try:
            return numpy.array(losses).mean(axis=0)
        except:  # parallel losses
            return numpy.array([x[:len(self.loss_weights) - 1]
                                for x in losses]).mean(axis=0)

    def eval_epoch(self):
        """
        Evaluate the network for one epoch and return the average loss.

        Returns:
            loss (float, list(float)): list of mean losses

        """
        self.model.eval()
        losses = []

        iterator = self._dataset_iterator(self.valid_loader)
        with torch.no_grad():
            for i_batch, batch in enumerate(iterator, 1):
                batch = self._batch_to_device(batch)

                batch_losses, batch_outputs = self._process_batch(*batch)

                # aggregate the losses into a single loss value
                loss, _losses = self._aggregate_losses(batch_losses)
                losses.append(_losses)

        return numpy.array(losses).mean(axis=0)

    def get_state(self):
        """
        Return a dictionary with the current state of the model.
        The state should contain all the important properties which will
        be save when taking a model checkpoint.
        Returns:
            state (dict)

        """
        state = {
            "config": self.config,
            "epoch": self.epoch,
            "step": self.step,
            "model": self.model.state_dict(),
            "model_class": self.model.__class__.__name__,
            "optimizers": [x.state_dict() for x in self.optimizers],
        }

        return state

    def checkpoint(self, name=None, timestamp=False, tags=None, verbose=False):

        if name is None:
            name = self.config["name"]

        return save_checkpoint(self.get_state(),
                               name=name, tag=tags, timestamp=timestamp,
                               verbose=verbose)
