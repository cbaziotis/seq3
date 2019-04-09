import os
from abc import ABC

from nltk import word_tokenize
from tabulate import tabulate
from torch.utils.data import Dataset

from modules.data.utils import vectorize, read_corpus, read_corpus_subw, \
    unks_per_sample, token_swaps


class BaseLMDataset(Dataset, ABC):
    def __init__(self, input, preprocess=None,
                 vocab=None, vocab_size=None,
                 subword=False, subword_path=None, verbose=True, **kwargs):
        """
        Base Dataset for Language Modeling.

        Args:
            preprocess (callable): preprocessing callable, which takes as input
                a string and returns a list of tokens
            input (str, list): the path to the data file, or a list of samples.
            vocab (Vocab): a vocab instance. If None, then build a new one
                from the Datasets data.
            vocab_size(int): if given, then trim the vocab to the given number.
            subword(bool): whether the dataset will be
                tokenized using subword units, using the SentencePiece package.
            subword(SentencePieceProcessor): path to the sentencepiece model
            verbose(bool): print useful statistics about the dataset.
        """
        self.input = input
        self.subword = subword
        self.subword_path = subword_path

        if preprocess is not None:
            self.preprocess = preprocess

        # tokenize the dataset
        if self.subword:
            self.vocab, self.data = read_corpus_subw(input, subword_path)
        else:
            self.vocab, self.data = read_corpus(input, self.preprocess)

        if vocab is not None:
            self.vocab = vocab
        else:
            self.vocab.build(vocab_size)

        if verbose:
            print(self)
            print()

    def __str__(self):

        props = []
        if isinstance(self.input, str):
            props.append(("source", os.path.basename(self.input)))

        _covarage = unks_per_sample(self.vocab.tok2id.keys(), self.data)
        _covarage = str(_covarage.round(4)) + " %"

        try:
            props.append(("size", len(self)))
        except:
            pass
        props.append(("vocab size", len(self.vocab)))
        props.append(("unique tokens", len(self.vocab.vocab)))
        props.append(("UNK per sample", _covarage))
        props.append(("subword", self.subword))

        if hasattr(self, 'seq_len'):
            props.append(("max seq length", self.seq_len))
        if hasattr(self, 'bptt'):
            props.append(("BPTT", self.bptt))
        if hasattr(self, 'attributes'):
            props.append(("attributes", len(self.attributes[0])))

        return tabulate([[x[1] for x in props]], headers=[x[0] for x in props])

    def truncate(self, n):
        self.data = self.data[:n]

    @staticmethod
    def preprocess(text, lower=True):
        if lower:
            text = text.lower()
        # return text.split()
        return word_tokenize(text)


class SentenceLMDataset(BaseLMDataset):
    def __init__(self, *args, seq_len=1000, **kwargs):
        """
        Dataset for sentence-level Language Modeling.
        """
        super().__init__(*args, **kwargs)
        # todo: find more elegant way to ignore seq_len
        self.seq_len = seq_len
        self.sos = kwargs.get("sos", False)
        self.oovs = kwargs.get("oovs", 0)

        for i in range(self.oovs):
            self.vocab.add_token(f"<oov-{i}>")
        print()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence = self.data[index]
        sentence = sentence + [self.vocab.EOS]

        if self.sos:
            sentence = [self.vocab.SOS] + sentence

        sentence = sentence[:self.seq_len]
        inputs = sentence[:-1]
        targets = sentence[1:]

        length = len(inputs)

        if self.oovs > 0:
            inputs_vec, _ = vectorize(inputs, self.vocab, self.oovs)
            targets_vec, _ = vectorize(targets, self.vocab, self.oovs)
        else:
            inputs_vec = vectorize(inputs, self.vocab)
            targets_vec = vectorize(targets, self.vocab)

        assert len(inputs_vec) == len(targets_vec)

        return inputs_vec, targets_vec, length


class AEDataset(BaseLMDataset):
    def __init__(self, *args, seq_len=250, **kwargs):
        """
        Dataset for sequence autoencoder.

        """
        super().__init__(*args, **kwargs)
        # todo: find more elegant way to ignore seq_len
        self.seq_len = seq_len
        self.oovs = kwargs.get("oovs", 0)
        self.return_oov = kwargs.get("return_oov", False)
        self.swaps = kwargs.get("swaps", 0.0)

        for i in range(self.oovs):
            self.vocab.add_token(f"<oov-{i}>")
        print()

    def __len__(self):
        return len(self.data)

    def read_sample(self, index):
        sample = self.data[index][:self.seq_len]
        sample = [self.vocab.SOS] + sample + [self.vocab.EOS]
        sample, _ = vectorize(sample, self.vocab, self.oovs)
        return list(map(self.vocab.id2tok.get, sample))

    def __getitem__(self, index):

        inp_x = self.data[index][:self.seq_len]
        out_x = inp_x[1:] + [self.vocab.EOS]

        inp_xhat = [self.vocab.SOS] + self.data[index][:self.seq_len]
        out_xhat = inp_xhat[1:] + [self.vocab.EOS]

        # print(tabulate([inp_src, out_src, inp_trg, out_trg],
        #                tablefmt="psql"))

        if not self.subword:
            inp_x, oov_map = vectorize(inp_x, self.vocab, self.oovs)
            out_x, _ = vectorize(out_x, self.vocab, self.oovs)
            inp_xhat, _ = vectorize(inp_xhat, self.vocab, self.oovs)
            out_xhat, _ = vectorize(out_xhat, self.vocab, self.oovs)
        else:
            raise NotImplementedError

        # add noise in the form of token swaps ! after the OOV replacements
        inp_x = token_swaps(inp_x, self.swaps)

        sample = inp_x, out_x, inp_xhat, out_xhat, len(inp_x), len(inp_xhat)

        if self.return_oov:
            sample = sample + (oov_map,)

        return sample
