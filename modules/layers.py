import torch
from torch import nn
from torch.autograd import Variable

from modules.helpers import sequence_mask, masked_normalization_inf


class GaussianNoise(nn.Module):
    def __init__(self, stddev, mean=.0):
        """
        Additive Gaussian Noise layer
        Args:
            stddev (float): the standard deviation of the distribution
            mean (float): the mean of the distribution
        """
        super().__init__()
        self.stddev = stddev
        self.mean = mean

    def forward(self, x):
        if self.training:
            # todo data_bug
            noise = Variable(x.data.new(x.size()).normal_(self.mean,
                                                          self.stddev))
            return x + noise
        return x

    def __repr__(self):
        return '{} (mean={}, stddev={})'.format(self.__class__.__name__,
                                                str(self.mean),
                                                str(self.stddev))


class Embed(nn.Module):
    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 embeddings=None,
                 noise=.0,
                 dropout=.0,
                 trainable=True, grad_mask=None, norm=False):
        """
        Define the layer of the model and perform the initializations
        of the layers (wherever it is necessary)
        Args:
            embeddings (numpy.ndarray): the 2D ndarray with the word vectors
            noise (float):
            dropout (float):
            trainable (bool):
        """
        super(Embed, self).__init__()

        self.norm = norm

        # define the embedding layer, with the corresponding dimensions
        self.embedding = nn.Embedding(num_embeddings=num_embeddings,
                                      embedding_dim=embedding_dim)

        # initialize the weights of the Embedding layer,
        # with the given pre-trained word vectors
        if embeddings is not None:
            print("Initializing Embedding layer with pre-trained weights!")
            self.init_embeddings(embeddings, trainable)

        # the dropout "layer" for the word embeddings
        self.dropout = nn.Dropout(dropout)

        # the gaussian noise "layer" for the word embeddings
        self.noise = GaussianNoise(noise)

        self.grad_mask = grad_mask

        if self.norm:
            self.layer_norm = nn.LayerNorm(embedding_dim)

        if self.grad_mask is not None:
            self.set_grad_mask(self.grad_mask)

    def _emb_hook(self, grad):
        return grad * Variable(self.grad_mask.unsqueeze(1)).type_as(grad)

    def set_grad_mask(self, mask):
        self.grad_mask = torch.from_numpy(mask)
        self.embedding.weight.register_hook(self._emb_hook)

    def init_embeddings(self, weights, trainable):
        self.embedding.weight = nn.Parameter(torch.from_numpy(weights),
                                             requires_grad=trainable)

    def regularize(self, embeddings):
        if self.noise.stddev > 0:
            embeddings = self.noise(embeddings)

        if self.dropout.p > 0:
            embeddings = self.dropout(embeddings)

        return embeddings

    def expectation(self, dists):
        """
        Obtain a weighted sum (expectation) of all the embeddings, from a
        given probability distribution.

        """
        flat_probs = dists.contiguous().view(dists.size(0) * dists.size(1),
                                             dists.size(2))
        flat_embs = flat_probs.mm(self.embedding.weight)
        embs = flat_embs.view(dists.size(0), dists.size(1), flat_embs.size(1))

        # apply layer normalization on the expectation
        if self.norm:
            embs = self.layer_norm(embs)

        # apply all embedding layer's regularizations
        embs = self.regularize(embs)

        return embs

    def forward(self, x):
        """
        This is the heart of the model. This function, defines how the data
        passes through the network.
        Args:
            x (): the input data (the sentences)

        Returns: the logits for each class

        """
        embeddings = self.embedding(x)

        if self.norm:
            embeddings = self.layer_norm(embeddings)

        embeddings = self.regularize(embeddings)

        return embeddings


class Attention(nn.Module):
    def __init__(self,
                 input_size,
                 context_size,
                 batch_first=True,
                 non_linearity="tanh",
                 method="general",
                 coverage=False):
        super(Attention, self).__init__()

        self.batch_first = batch_first
        self.method = method
        self.coverage = coverage

        if self.method not in ["dot", "general", "concat", "additive"]:
            raise ValueError("Please select a valid attention type.")

        if self.coverage:
            self.W_c = nn.Linear(1, context_size, bias=False)
            self.method = "additive"

        if non_linearity == "relu":
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Tanh()

        if self.method == "general":
            self.W_h = nn.Linear(input_size, context_size)

        elif self.method == "additive":
            self.W_h = nn.Linear(input_size, context_size)
            self.W_s = nn.Linear(context_size, context_size)
            self.W_v = nn.Linear(context_size, 1)

        elif self.method == "concat":
            self.W_h = nn.Linear(input_size + context_size, context_size)
            self.W_v = nn.Linear(context_size, 1)

    def score(self, sequence, query, coverage=None):
        batch_size, max_length, feat_size = sequence.size()

        if self.method == "dot":
            energies = torch.matmul(sequence, query.unsqueeze(2)).squeeze(2)

        elif self.method == "additive":
            enc = self.W_h(sequence)
            dec = self.W_s(query)
            sums = enc + dec.unsqueeze(1)

            if self.coverage:
                cov = self.W_c(coverage.unsqueeze(-1))
                sums = sums + cov

            energies = self.W_v(self.activation(sums)).squeeze(2)

        elif self.method == "general":
            h = self.W_h(sequence)
            energies = torch.matmul(h, query.unsqueeze(2)).squeeze(2)

        elif self.method == "concat":
            c = query.unsqueeze(1).expand(-1, max_length, -1)
            u = self.W_h(torch.cat([sequence, c], -1))
            energies = self.W_v(self.activation(u)).squeeze(2)

        else:
            raise ValueError

        return energies

    def forward(self, sequence, query, lengths, coverage=None):

        energies = self.score(sequence, query, coverage)

        # construct a mask, based on sentence lengths
        mask = sequence_mask(lengths, energies.size(1))

        scores = masked_normalization_inf(energies, mask)
        # scores = self.masked_normalization(energies, mask)

        contexts = (sequence * scores.unsqueeze(-1)).sum(1)

        return contexts, scores
