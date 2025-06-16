#!/bin/bash

mkdir -p logs

nohup python3 experiments_Mario.py --NUM_OF_SYMBOLS=5 --NUM_OF_STATES=25 --DATASET_DIR="observation_clean_labels_10" --NUM_LABELS=10 > ./logs/NUM_LABELS_10/log_Symbols_5_States_25.log 2>&1 &
nohup python3 experiments_Mario.py --NUM_OF_SYMBOLS=10 --NUM_OF_STATES=25 --DATASET_DIR="observation_clean_labels_10" --NUM_LABELS=10 > ./logs/NUM_LABELS_10/log_Symbols_10_States_25.log 2>&1 &
nohup python3 experiments_Mario.py --NUM_OF_SYMBOLS=15 --NUM_OF_STATES=25 --DATASET_DIR="observation_clean_labels_10" --NUM_LABELS=10 > ./logs/NUM_LABELS_10/log_Symbols_15_States_25.log 2>&1 &
nohup python3 experiments_Mario.py --NUM_OF_SYMBOLS=20 --NUM_OF_STATES=25 --DATASET_DIR="observation_clean_labels_10" --NUM_LABELS=10 > ./logs/NUM_LABELS_10/log_Symbols_20_States_25.log 2>&1 &

nohup python3 experiments_Mario.py --NUM_OF_SYMBOLS=5 --NUM_OF_STATES=50 --DATASET_DIR="observation_clean_labels_10" --NUM_LABELS=10 > ./logs/NUM_LABELS_10/log_Symbols_5_States_50.log 2>&1 &
nohup python3 experiments_Mario.py --NUM_OF_SYMBOLS=10 --NUM_OF_STATES=50 --DATASET_DIR="observation_clean_labels_10" --NUM_LABELS=10 > ./logs/NUM_LABELS_10/log_Symbols_10_States_50.log 2>&1 &
nohup python3 experiments_Mario.py --NUM_OF_SYMBOLS=15 --NUM_OF_STATES=50 --DATASET_DIR="observation_clean_labels_10" --NUM_LABELS=10 > ./logs/NUM_LABELS_10/log_Symbols_15_States_50.log 2>&1 &
nohup python3 experiments_Mario.py --NUM_OF_SYMBOLS=20 --NUM_OF_STATES=50 --DATASET_DIR="observation_clean_labels_10" --NUM_LABELS=10 > ./logs/NUM_LABELS_10/log_Symbols_20_States_50.log 2>&1 &




