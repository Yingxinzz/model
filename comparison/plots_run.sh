#!/bin/bash
# run.sh - Run visualize.py for all possible dataset options

SCRIPT="plots_benchmark.py"

echo " Running plots_benchmark.py for all possible datasets"

python $SCRIPT -d dlpfc_sample1
python $SCRIPT -d dlpfc_sample2
python $SCRIPT -d dlpfc_sample3
python $SCRIPT -d dlpfc_7374
python $SCRIPT -d dlpfc_all
python $SCRIPT -d hbc
python $SCRIPT -d coronal
python $SCRIPT -d mob

python $SCRIPT -d check

echo " All plots_benchmark.py runs completed!"
