import math

import numpy
import torch
from torch.utils.data import Sampler


class BPTTSampler(Sampler):
    """
    Samples elements per chunk. Suitable for Language Models.
    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, size, batch):
        """
        Define how to construct batches

        Given a list of sequences, organize the sequences in each batch
        in such a way, so that each RNN gets the proper (next) sequence.

        For example, given the following sequence and with batch=2:
        ┌ a b c d e ┐
        │ f g h i j │
        │ k l m n o │
        │ p q r s t │
        │ u v w x y │
        └ z - - - - ┘

        the batches will be:
        ┌ a b c d e ┐    ┌ f g h i j ┐    ┌ k l m n o ┐
        └ p q r s t ┘    └ u v w x y ┘    └ z - - - - ┘

        Args:
            size (int): number of sequences
            batch (int): batch size
        """
        self.size = size
        self.batch = batch

        # split the corpus in chunks of size `corpus_seqs / batch_size`
        self.chunks = numpy.array_split(numpy.arange(self.size), batch)

    def get_batch(self, index):
        """
        Fill each batch with the i-th sequence from each chunk.
        If the batch size does not evenly divides the chunks,
        then some chunks will have one less sequence, so the last batch
        will have fewer samples.
        Args:
            index (int):

        Returns:

        """
        batch = []
        for chunk in self.chunks:
            if index < chunk.size:
                batch.append(chunk[index])
        return batch

    def batches(self):
        for i in range(self.chunks[0].size):
            yield self.get_batch(i)

    def __iter__(self):
        return iter(self.batches())

    def __len__(self):
        return self.size


def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]


class SortedSampler(Sampler):
    """
    Defines a strategy for drawing samples from the dataset,
    in ascending or descending order, based in the sample lengths.
    """

    def __init__(self, lengths, descending=False):
        self.lengths = lengths
        self.desc = descending

    def __iter__(self):

        if self.desc:
            return iter(numpy.flip(numpy.array(self.lengths).argsort(), 0))
        else:
            return iter(numpy.array(self.lengths).argsort())

    def __len__(self):
        return len(self.lengths)


class BucketBatchSampler(Sampler):
    """
    Defines a strategy for drawing batches of samples from the dataset,
    in ascending or descending order, based in the sample lengths.
    """

    def __init__(self, lengths, batch_size,
                 shuffle=False, even=False, drop_last=False, reverse=False):
        sorted_indices = numpy.array(lengths).argsort()
        num_sections = math.ceil(len(lengths) / batch_size)
        if even:
            self.batches = list(divide_chunks(sorted_indices, batch_size))
        else:
            self.batches = numpy.array_split(sorted_indices, num_sections)

        if reverse:
            self.batches = list(reversed(self.batches))

        if drop_last:
            del self.batches[-1]

        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            return iter(self.batches[i]
                        for i in torch.randperm(len(self.batches)))
        else:
            return iter(self.batches)

    def __len__(self):
        return len(self.batches)
