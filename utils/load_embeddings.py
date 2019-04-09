import errno
import os
import pickle

import numpy


def file_cache_name(file):
    head, tail = os.path.split(file)
    filename, ext = os.path.splitext(tail)
    return os.path.join(head, filename + ".p")


def write_cache_word_vectors(file, data):
    with open(file_cache_name(file), 'wb') as pickle_file:
        pickle.dump(data, pickle_file)


def load_cache_word_vectors(file):
    with open(file_cache_name(file), 'rb') as f:
        return pickle.load(f)


def load_word_vectors(file, dim):
    """
    Read the word vectors from a text file
    Args:
        file (): the filename
        dim (): the dimensions of the word vectors

    Returns:
        word2idx (dict): dictionary of words to ids
        idx2word (dict): dictionary of ids to words
        embeddings (numpy.ndarray): the word embeddings matrix

    """
    # in order to avoid this time consuming operation, cache the results
    try:
        cache = load_cache_word_vectors(file)
        print("Loaded word embeddings from cache.")
        return cache
    except OSError:
        print("Didn't find embeddings cache file {}".format(file))

    # create the necessary dictionaries and the word embeddings matrix
    if os.path.exists(file):
        print('Indexing file {} ...'.format(file))

        word2idx = {}  # dictionary of words to ids
        idx2word = {}  # dictionary of ids to words
        embeddings = []  # the word embeddings matrix

        # create the 2D array, which will be used for initializing
        # the Embedding layer of a NN.
        # We reserve the first row (idx=0), as the word embedding,
        # which will be used for zero padding (word with id = 0).
        embeddings.append(numpy.zeros(dim))

        # flag indicating whether the first row of the embeddings file
        # has a header
        header = False

        # read file, line by line
        with open(file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):

                # skip the first row if it is a header
                if i == 1:
                    if len(line.split()) < dim:
                        header = True
                        continue

                values = line.split(" ")
                word = values[0]
                vector = numpy.asarray(values[1:], dtype='float32')

                index = i - 1 if header else i

                idx2word[index] = word
                word2idx[word] = index
                embeddings.append(vector)

            # add an unk token, for OOV words
            if "<unk>" not in word2idx:
                idx2word[len(idx2word) + 1] = "<unk>"
                word2idx["<unk>"] = len(word2idx) + 1
                embeddings.append(
                    numpy.random.uniform(low=-0.05, high=0.05, size=dim))

            print(set([len(x) for x in embeddings]))

            print('Found %s word vectors.' % len(embeddings))
            embeddings = numpy.array(embeddings, dtype='float32')

        # write the data to a cache file
        write_cache_word_vectors(file, (word2idx, idx2word, embeddings))

        return word2idx, idx2word, embeddings

    else:
        print("{} not found!".format(file))
        raise OSError(errno.ENOENT, os.strerror(errno.ENOENT), file)
