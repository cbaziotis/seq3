def giga_tokenizer(x):
    return x.split()


with open("input.txt") as inp, open("f8w.txt", "w") as out:
    for line in inp:
        f8w = giga_tokenizer(line)[:8]
        out.write(" ".join(f8w) + "\n")


with open("input_filtered.txt") as inp, open("f8w_filtered.txt", "w") as out:
    for line in inp:
        f8w = giga_tokenizer(line)[:8]
        out.write(" ".join(f8w) + "\n")

with open("input_min8.txt") as inp, open("f8w_min8.txt", "w") as out:
    for line in inp:
        f8w = giga_tokenizer(line)[:8]
        out.write(" ".join(f8w) + "\n")
