from glob import glob

from numpy import mean


def giga_tokenizer(x):
    return x.strip().lower().split()


def count_dataset_lengths(file):
    lengths = []

    with open(file) as f:
        for line in f:
            # lengths.append(len(giga_tokenizer(line)))
            lengths.append(len(line.split()))

    return lengths


for f in glob('*.txt'):
    avg_length = mean(count_dataset_lengths(f))
    print(f"{f}: {avg_length}")
