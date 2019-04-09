#!/usr/bin/env bash

files2rouge hyps/DUC2003_full-pos_preds.txt    ./DUC2003/all_refs.txt -a "-b 75 -m -n 2 -w 1.2 -a" -e "<eos>" -s ./results/DUC2003_full-pos_results.txt
files2rouge hyps/DUC2003_full-prior_preds.txt  ./DUC2003/all_refs.txt -a "-b 75 -m -n 2 -w 1.2 -a" -e "<eos>" -s ./results/DUC2003_full-prior_results.txt
files2rouge hyps/DUC2003_full-topic_preds.txt  ./DUC2003/all_refs.txt -a "-b 75 -m -n 2 -w 1.2 -a" -e "<eos>" -s ./results/DUC2003_full-topic_results.txt
files2rouge hyps/DUC2003_full_preds.txt        ./DUC2003/all_refs.txt -a "-b 75 -m -n 2 -w 1.2 -a" -e "<eos>" -s ./results/DUC2003_full_results.txt
files2rouge hyps/DUC2004_full-pos_preds.txt    ./DUC2004/all_refs.txt -a "-b 75 -m -n 2 -w 1.2 -a" -e "<eos>" -s ./results/DUC2004_full-pos_results.txt
files2rouge hyps/DUC2004_full-prior_preds.txt  ./DUC2004/all_refs.txt -a "-b 75 -m -n 2 -w 1.2 -a" -e "<eos>" -s ./results/DUC2004_full-prior_results.txt
files2rouge hyps/DUC2004_full-topic_preds.txt  ./DUC2004/all_refs.txt -a "-b 75 -m -n 2 -w 1.2 -a" -e "<eos>" -s ./results/DUC2004_full-topic_results.txt
files2rouge hyps/DUC2004_full_preds.txt        ./DUC2004/all_refs.txt -a "-b 75 -m -n 2 -w 1.2 -a" -e "<eos>" -s ./results/DUC2004_full_results.txt









files2rouge hyps/DUC2004_full20K_preds.txt    ./DUC2004/all_refs.txt -a "-b 75 -m -n 2 -w 1.2 -a" -e "<eos>" -s ./results/DUC2004_full20K_results.txt
