from modules.training.trainer import Trainer


class LMTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _seq_loss(self, predictions, labels):
        _labels = labels.contiguous().view(-1)

        _logits = predictions[0]
        _logits = _logits.contiguous().view(-1, _logits.size(-1))
        loss = self.criterion(_logits, _labels)

        return loss

    def _process_batch(self, inputs, labels, lengths):
        predictions = self.model(inputs, None, lengths)

        loss = self._seq_loss(predictions, labels)
        del predictions
        predictions = None

        return loss, predictions

    def get_state(self):
        if self.train_loader.dataset.subword:
            _vocab = self.train_loader.dataset.subword_path
        else:
            _vocab = self.train_loader.dataset.vocab

        state = {
            "config": self.config,
            "epoch": self.epoch,
            "step": self.step,
            "model": self.model.state_dict(),
            "model_class": self.model.__class__.__name__,
            "optimizers": [x.state_dict() for x in self.optimizers],
            "vocab": _vocab,
        }

        return state
