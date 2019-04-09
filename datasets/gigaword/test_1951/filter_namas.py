def giga_tokenizer(x):
    return x.strip().lower().split()


def filter_namas(title, article):
    title_words = title.split()
    article_words = article.split()


    # Reasonable lengths
    if not (10 < len(article_words) < 100 and
            3 < len(title_words) < 50):
        return False

    return True


src = open("./input.txt").readlines()
trg = open("./task1_ref0.txt").readlines()

with open("./input_NAMAS.txt", "w") as f1, \
        open("./task1_ref0_NAMAS.txt", "w") as f2:
    for s, t in zip(src, trg):
        if filter_namas(s, t):
            f1.write(s)
            f2.write(t)
