import torch
from torch.nn.utils.rnn import pad_sequence


class SeqCollate:
    """
    Base Class.
    A variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self, sort=False, batch_first=True):
        self.sort = sort
        self.batch_first = batch_first

    def pad_samples(self, samples):
        return pad_sequence([torch.LongTensor(x) for x in samples],
                            self.batch_first)

    def _collate(self, *args):
        raise NotImplementedError

    def __call__(self, batch):
        batch = list(zip(*batch))
        return self._collate(*batch)


class LMCollate(SeqCollate):
    def __init__(self, *args):
        super().__init__(*args)

    def _collate(self, inputs, targets, lengths):
        inputs = self.pad_samples(inputs)
        targets = self.pad_samples(targets)
        lengths = torch.LongTensor(lengths)
        return inputs, targets, lengths


class CondLMCollate(SeqCollate):
    def __init__(self, *args):
        super().__init__(*args)

    def _collate(self, inputs, targets, attributes, lengths):
        inputs = self.pad_samples(inputs)
        targets = self.pad_samples(targets)
        attributes = self.pad_samples(attributes)
        lengths = torch.LongTensor(lengths)
        return inputs, targets, attributes, lengths


class Seq2SeqCollate(SeqCollate):
    def __init__(self, *args):
        super().__init__(*args)

    def _collate(self, inp_src, out_src, inp_trg, out_trg, len_src, len_trg):
        inp_src = self.pad_samples(inp_src)
        out_src = self.pad_samples(out_src)
        inp_trg = self.pad_samples(inp_trg)
        out_trg = self.pad_samples(out_trg)

        len_src = torch.LongTensor(len_src)
        len_trg = torch.LongTensor(len_trg)

        return inp_src, out_src, inp_trg, out_trg, len_src, len_trg


class Seq2SeqOOVCollate(SeqCollate):
    def __init__(self, *args):
        super().__init__(*args)

    def _collate(self, inp_src, out_src, inp_trg, out_trg, len_src, len_trg,
                 oov_map):
        inp_src = self.pad_samples(inp_src)
        out_src = self.pad_samples(out_src)
        inp_trg = self.pad_samples(inp_trg)
        out_trg = self.pad_samples(out_trg)

        len_src = torch.LongTensor(len_src)
        len_trg = torch.LongTensor(len_trg)

        return inp_src, out_src, inp_trg, out_trg, len_src, len_trg, oov_map
