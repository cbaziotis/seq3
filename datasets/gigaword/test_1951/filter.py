def giga_tokenizer(x):
    return x.strip().lower().split()


src = open("./input.txt").readlines()
trg = open("./task1_ref0.txt").readlines()

with open("./input_min8.txt", "w") as f1, \
        open("./task1_ref0_min8.txt", "w") as f2:
    for s, t in zip(src, trg):
        if len(giga_tokenizer(s)) > 8:
            f1.write(s)
            f2.write(t)

