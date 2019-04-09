import os
import random

from sys_config import DATA_DIR
from utils.generic import number_h

size = 4000
h_size = number_h(size).replace('.0', '')

src_file = os.path.join(DATA_DIR, "gigaword/valid.article.filter.txt")
trg_file = os.path.join(DATA_DIR, "gigaword/valid.title.filter.txt")

src_filename, src_ext = os.path.splitext(src_file)
trg_filename, trg_ext = os.path.splitext(trg_file)

src_subset_file = src_filename + f".{h_size}" + src_ext
trg_subset_file = trg_filename + f".{h_size}" + trg_ext

src_data = open(src_file).readlines()
trg_data = open(trg_file).readlines()

dataset = list(zip(src_data, trg_data))
random.shuffle(dataset)

with open(src_subset_file, "w") as f1, open(trg_subset_file, "w") as f2:
    for s, t in dataset[:size]:
        f1.write(s)
        f2.write(t)
