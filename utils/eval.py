import pandas

import rouge
from tabulate import tabulate


def rouge_lists(refs, hyps):
    evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l'],
                            max_n=2,
                            limit_length=True,
                            length_limit=100,
                            length_limit_type='words',
                            apply_avg=True,
                            apply_best=False,
                            alpha=0.5,  # Default F1_score
                            weight_factor=1.2,
                            stemming=True)
    scores = evaluator.get_scores(hyps, refs)

    return scores


def rouge_files(refs_file, hyps_file):
    refs = open(refs_file).readlines()
    hyps = open(hyps_file).readlines()
    scores = rouge_lists(refs, hyps)

    return scores


def rouge_file_list(refs_file, hyps_list):
    refs = open(refs_file).readlines()
    scores = rouge_lists(refs, hyps_list)

    return scores


def pprint_rouge_scores(scores, pivot=False):
    pdt = pandas.DataFrame(scores)

    if pivot:
        pdt = pdt.T

    table = tabulate(pdt,
                     headers='keys',
                     floatfmt=".4f", tablefmt="psql")

    return table
