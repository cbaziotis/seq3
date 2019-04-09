from pprint import pprint

import torch
from sklearn.feature_extraction.text import TfidfVectorizer

from generate.utils import devectorize


def compute_dataset_idf(dataset, vocab):
    def identity_func(doc):
        return doc

    my_data = [dataset.read_sample(i) for i in range(len(dataset))]

    tf = TfidfVectorizer(lowercase=False,
                         tokenizer=identity_func,
                         preprocessor=identity_func,
                         use_idf=True,
                         vocabulary=vocab)
    tf.fit(my_data)
    return tf.idf_


def sample2text(word_ids, vocab):
    tokens = devectorize(word_ids,
                         vocab.id2tok,
                         vocab.tok2id[vocab.EOS],
                         strip_eos=False,
                         pp=False)
    text = [" ".join(out) for out in tokens]
    lengths = [len(t) for t in tokens]
    return text, lengths


def str2tree(ls):
    tree = {}
    for item in ls:
        t = tree
        for part in item.split('.'):
            t = t.setdefault(part, {})
    pprint(tree)


def sample_lengths(src_lengths,
                   min_ratio, max_ratio,
                   min_length, max_length):
    """
    Sample summary lengths from a list of source lengths.

    """
    t = torch.empty(len(src_lengths), device=src_lengths.device)
    samples = t.uniform_(min_ratio, max_ratio)
    lengths = (src_lengths.float() * samples).long()
    lengths = lengths.clamp(min=min_length, max=max_length)
    return lengths
