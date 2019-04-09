import re


def giga_tokenizer(x):
    return x.split()


base = "DUC2004"

task1_ref0 = open(base + "/task1_ref0.txt").readlines()
task1_ref1 = open(base + "/task1_ref1.txt").readlines()
task1_ref2 = open(base + "/task1_ref2.txt").readlines()
task1_ref3 = open(base + "/task1_ref3.txt").readlines()


def preprocess_line(line):
    return " ".join(re.sub(r'\d', '#', line.lower().strip()).split())


with open(base + "/all_refs.txt", "w") as f:
    for t1, t2, t3, t4 in zip(task1_ref0, task1_ref1, task1_ref2, task1_ref3):
        pt1 = preprocess_line(t1)
        pt2 = preprocess_line(t2)
        pt3 = preprocess_line(t3)
        pt4 = preprocess_line(t4)

        f.write("<eos>".join([pt1, pt2, pt3, pt4]) + "\n")
