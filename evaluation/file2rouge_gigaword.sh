#!/usr/bin/env bash
#files2rouge /home/christos/PycharmProjects/seq2seq2seq/datasets/gigaword/test_1951/f8w.txt /home/christos/PycharmProjects/seq2seq2seq/datasets/gigaword/test_1951/task1_ref0.txt -a " -m -n 2 -w 1.2 -a"
#files2rouge /home/christos/PycharmProjects/seq2seq2seq/datasets/gigaword/test_1951/f8w_min8.txt /home/christos/PycharmProjects/seq2seq2seq/datasets/gigaword/test_1951/task1_ref0_min8.txt -a " -m -n 2 -w 1.2 -a"
#files2rouge /home/christos/PycharmProjects/seq2seq2seq/datasets/gigaword/test_1951/f8w_filtered.txt /home/christos/PycharmProjects/seq2seq2seq/datasets/gigaword/test_1951/task1_ref0_filtered.txt -a " -m -n 2 -w 1.2 -a"
#


files2rouge ./hyps/gigaword_full_preds.txt ./gigaword/task1_ref0_min8.txt -a " -m -n 2 -w 1.2 -a" -s ./results/gigaword_full_results.txt
files2rouge ./hyps/gigaword_full-prior_preds.txt ./gigaword/task1_ref0_min8.txt -a " -m -n 2 -w 1.2 -a" -s ./results/gigaword_full-prior_results.txt
files2rouge ./hyps/gigaword_full-topic_preds.txt ./gigaword/task1_ref0_min8.txt -a " -m -n 2 -w 1.2 -a" -s ./results/gigaword_full-topic_results.txt

files2rouge ./hyps/gigaword_full20K_preds.txt ./gigaword/task1_ref0_min8.txt -a " -m -n 2 -w 1.2 -a" -s ./results/gigaword_full20K_results.txt
