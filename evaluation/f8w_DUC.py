import re


def giga_tokenizer(x):
    return x.split()


base = "DUC2003"

with open(base + "/input.txt") as inp, \
        open(base + "/prefix.txt", "w") as pf, open(
    base + "/lead8.txt", "w") as f8:
    for line in inp:
        preprocessed = re.sub(r'\d', '#', line.lower().strip())

        prefix = preprocessed[:75]
        lead8 = " ".join(preprocessed.split()[:8])

        pf.write(prefix + "\n")

        f8.write(lead8 + "\n")
