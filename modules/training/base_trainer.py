import math

import numpy
import torch


class BaseTrainer:
    def __init__(self, train_loader, valid_loader,
                 config, device,
                 batch_end_callbacks=None, loss_weights=None,
                 parallel=False,
                 **kwargs):

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device
        self.loss_weights = loss_weights

        self.config = config
        self.log_interval = self.config["log_interval"]
        self.batch_size = self.config["batch_size"]
        self.checkpoint_interval = self.config["checkpoint_interval"]
        self.clip = self.config["model"]["clip"]

        if batch_end_callbacks is None:
            self.batch_end_callbacks = []
        else:
            self.batch_end_callbacks = [c for c in batch_end_callbacks
                                        if callable(c)]

        self.epoch = 0
        self.step = 0
        self.progress_log = None

        # init dataset
        self.train_set_size = self._get_dataset_size(self.train_loader)
        self.val_set_size = self._get_dataset_size(self.valid_loader)

        self.n_batches = math.ceil(
            float(self.train_set_size) / self.batch_size)
        self.total_steps = self.n_batches * self.config["epochs"]

        if self.loss_weights is not None:
            self.loss_weights = [self.anneal_init(w) for w in
                                 self.loss_weights]

    @staticmethod
    def _roll_seq(x, dim=1, shift=1):
        length = x.size(dim) - shift

        seq = torch.cat([x.narrow(dim, shift, length),
                         torch.zeros_like(x[:, :1])], dim)

        return seq

    @staticmethod
    def _get_dataset_size(loader):
        """
        If the trainer holds multiple datasets, then the size
        is estimated based on the largest one.
        """
        if isinstance(loader, (tuple, list)):
            return len(loader[0].dataset)
        else:
            return len(loader.dataset)

    def anneal_init(self, param, steps=None):
        if isinstance(param, list):
            if steps is None:
                steps = self.total_steps
            return numpy.geomspace(param[0], param[1], num=steps).tolist()
        else:
            return param

    def anneal_step(self, param):
        if isinstance(param, list):
            try:
                _val = param[self.step]
            except:
                _val = param[-1]
        else:
            _val = param

        return _val

    def _tensors_to_device(self, batch):
        return list(map(lambda x: x.to(self.device), batch))

    def _batch_to_device(self, batch):

        if torch.is_tensor(batch[0]):
            batch = self._tensors_to_device(batch)
        else:
            batch = list(map(lambda x: self._tensors_to_device(x), batch))

        return batch

    @staticmethod
    def _multi_dataset_iter(loader, strategy, step=1):
        # todo: generalize to N datasets. For now works only with 2.
        sizes = [len(x) for x in loader]

        iter_a = iter(loader[0])
        iter_b = iter(loader[1])

        if strategy == "spread":
            step = math.floor((sizes[0] - sizes[1]) / (sizes[1] - 1))

            for i in range(max(sizes)):
                if i % (step + 1) == 0:
                    batch_a = next(iter_a)
                    batch_b = next(iter_b, None)

                    if batch_b is not None:
                        yield batch_a, batch_b
                    else:
                        yield batch_a
                else:
                    yield next(iter_a)

        if strategy == "modulo":
            for i in range(max(sizes)):
                if i % step == 0:
                    batch_a = next(iter_a)
                    batch_b = next(iter_b, None)

                    if batch_b is None:  # reset iterator b
                        iter_b = iter(loader[1])
                        batch_b = next(iter_b, None)

                    yield batch_a, batch_b
                else:
                    yield next(iter_a)

        elif strategy == "cycle":
            for i in range(max(sizes)):
                batch_a = next(iter_a)
                batch_b = next(iter_b, None)

                if batch_b is None:  # reset iterator b
                    iter_b = iter(loader[1])
                    batch_b = next(iter_b, None)

                yield batch_a, batch_b

        elif strategy == "beginning":
            for i in range(max(sizes)):
                batch_a = next(iter_a)
                batch_b = next(iter_b, None)

                if batch_b is not None:
                    yield batch_a, batch_b
                else:
                    yield batch_a
        else:
            raise ValueError("Invalid iteration strategy!")

    def _dataset_iterator(self, loader, strategy=None, step=1):
        # if all datasets have the same size
        if isinstance(loader, (tuple, list)):
            if len(set(len(x) for x in loader)) == 1:
                return zip(*loader)
            else:
                return self._multi_dataset_iter(loader, strategy, step)
        else:
            return loader

    def _aggregate_losses(self, batch_losses, loss_weights=None):
        """
        This function computes a weighted sum of the models losses
        Args:
            batch_losses(torch.Tensor, tuple):

        Returns:
            loss_sum (int): the aggregation of the constituent losses
            loss_list (list, int): the constituent losses

        """
        if isinstance(batch_losses, (tuple, list)):

            if loss_weights is None:
                loss_weights = self.loss_weights
                loss_weights = [self.anneal_step(w) for w in loss_weights]

            if loss_weights is None:
                loss_sum = sum(batch_losses)
                loss_list = [x.item() for x in batch_losses]
            else:
                loss_sum = sum(w * x for x, w in
                               zip(batch_losses, loss_weights))

                loss_list = [w * x.item() for x, w in
                             zip(batch_losses, loss_weights)]
        else:
            loss_sum = batch_losses
            loss_list = batch_losses.item()
        return loss_sum, loss_list
