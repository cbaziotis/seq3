from collections.__init__ import Counter

import numpy
from gensim.models import FastText
from tqdm import tqdm

from utils.load_embeddings import load_word_vectors


class Vocab(object):
    """
    The Vocab Class, holds the vocabulary of a corpus and
    mappings from tokens to indices and vice versa.
    """

    def __init__(self, pad="<pad>", sos="<sos>", eos="<eos>", unk="<unk>",
                 oovs=0):
        self.PAD = pad
        self.SOS = sos
        self.EOS = eos
        self.UNK = unk
        self.oovs = oovs

        self.vocab = Counter()

        self.tok2id = dict()
        self.id2tok = dict()

        self.size = 0

        self.subword = None

    def read_sequence(self, tokens):
        self.vocab.update(tokens)

    def trim(self, size):
        self.tok2id = dict()
        self.id2tok = dict()
        self.build(size)

    def read_embeddings(self, file, dim):
        """
        Create an Embeddings Matrix, in which each row corresponds to
        the word vector from the pretrained word embeddings.
        If a word is missing from the provided pretrained word vectors, then
        sample a new embedding, from the gaussian of the pretrained embeddings.

        Args:
            file:
            dim:

        Returns:

        """
        word2idx, idx2word, embeddings = load_word_vectors(file, dim)

        mu = embeddings.mean(axis=0)
        sigma = embeddings.std(axis=0)

        filtered_embeddings = numpy.zeros((len(self), embeddings.shape[1]))

        mask = numpy.zeros(len(self))
        missing = []

        for token_id, token in tqdm(self.id2tok.items(),
                                    desc="Reading embeddings...",
                                    total=len(self.id2tok.items())):
            if token not in word2idx or token == "<unk>":
                # todo: smart sampling per dim distribution
                # sample = numpy.random.uniform(low=-0.5, high=0.5,
                #                               size=embeddings.shape[1])
                sample = numpy.random.normal(mu, sigma / 4)
                filtered_embeddings[token_id] = sample

                mask[token_id] = 1
                missing.append(token_id)
            else:
                filtered_embeddings[token_id] = embeddings[word2idx[token]]

        print(f"Missing tokens from the pretrained embeddings: {len(missing)}")

        return filtered_embeddings, mask, missing

    def read_fasttext(self, file):
        """
        Create an Embeddings Matrix, in which each row corresponds to
        the word vector from the pretrained word embeddings.
        If a word is missing then obtain a representation on-the-fly
        using fasttext.

        Args:
            file:
            dim:

        Returns:

        """
        model = FastText.load_fasttext_format(file)

        embeddings = numpy.zeros((len(self), model.vector_size))

        missing = []

        for token_id, token in tqdm(self.id2tok.items(),
                                    desc="Reading embeddings...",
                                    total=len(self.id2tok.items())):
            if token not in model.wv.vocab:
                missing.append(token)
            embeddings[token_id] = model[token]

        print(f"Missing tokens from the pretrained embeddings: {len(missing)}")

        return embeddings, missing

    def add_token(self, token):
        index = len(self.tok2id)

        if token not in self.tok2id:
            self.tok2id[token] = index
            self.id2tok[index] = token
            self.size = len(self)

    def __add_special_tokens(self):
        self.add_token(self.PAD)
        self.add_token(self.SOS)
        self.add_token(self.EOS)
        self.add_token(self.UNK)

    def from_file(self, file, skip=0):
        self.__add_special_tokens()

        lines = open(file).readlines()[skip:]
        for line in lines:
            token = line.split()[0]
            self.add_token(token)

    def to_file(self, file):
        with open(file, "w") as f:
            f.write("\n".join(self.tok2id.keys()))

    def is_corrupt(self):
        return len([tok for tok, index in self.tok2id.items()
                    if self.id2tok[index] != tok]) > 0

    def get_tokens(self):
        return [self.id2tok[key] for key in sorted(self.id2tok.keys())]

    def build(self, size=None):
        self.__add_special_tokens()

        for w, k in self.vocab.most_common(size):
            self.add_token(w)

    def __len__(self):
        return len(self.tok2id)
