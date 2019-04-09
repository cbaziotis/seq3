import math

import numpy
import torch
from torch.utils.data import Sampler


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
