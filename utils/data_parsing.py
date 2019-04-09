import csv
import json
import os
import random
from collections import Counter, defaultdict
from matplotlib import pyplot as plt

import numpy
from glob2 import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from sys_config import DATA_DIR


def read_amazon(file):
    reviews = []
    summaries = []
    labels = []

    with open(file) as f:
        for line in f:
            entry = json.loads(line)
            reviews.append(entry["reviewText"])
            summaries.append(entry["summary"])
            labels.append(entry["overall"])

    return reviews, summaries, labels


def read_semeval():
    def read_dataset(d):
        with open(os.path.join(DATA_DIR, "semeval", "E-c",
                               "E-c-En-{}.txt".format(d))) as f:
            reader = csv.reader(f, delimiter='\t')
            labels = next(reader)[2:]

            _X = []
            _y = []
            for row in reader:
                _X.append(row[1])
                _y.append([int(x) for x in row[2:]])
            return _X, _y

    X_train, y_train = read_dataset("train")
    X_dev, y_dev = read_dataset("dev")
    X_test, y_test = read_dataset("test")

    X_train = X_train + X_test
    y_train = y_train + y_test

    return X_train, numpy.array(y_train), X_dev, numpy.array(y_dev)


def imdb_get_index():
    index = defaultdict(list)

    dirs = ["pos", "neg", "unsup"]
    sets = ["train", "test"]

    for s in sets:
        for d in dirs:
            for file in glob(os.path.join(DATA_DIR, "imdb", s, d) + "/*.txt"):
                index["_".join([s, d])].append(file)
    return index


def get_imdb():
    index = imdb_get_index()

    data = []

    for ki, vi in index.items():
        for f in vi:
            data.append(" ".join(open(f).readlines()).replace('<br />', ''))

    return data


def read_emoji(split=0.1, min_freq=100, max_ex=1000000, top_n=None):
    X = []
    y = []
    with open(os.path.join(DATA_DIR, "emoji", "emoji_1m.txt")) as f:
        for i, line in enumerate(f):
            if i > max_ex:
                break
            emoji, text = line.rstrip().split("\t")
            X.append(text)
            y.append(emoji)

    counter = Counter(y)
    top = set(l for l, f in counter.most_common(top_n) if f > min_freq)

    data = [(_x, _y) for _x, _y in zip(X, y) if _y in top]

    total = len(data)

    data = [(_x, _y) for _x, _y in data if
            random.random() > counter[_y] / total]

    X = [x[0] for x in data]
    y = [x[1] for x in data]

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=split,
                                                        stratify=y,
                                                        random_state=0)

    lb = LabelBinarizer()
    lb.fit(y_train)
    y_train = lb.transform(y_train)
    y_test = lb.transform(y_test)

    return X_train, y_train, X_test, y_test
