#!/bin/bash

PYTHON=python3 

# Fixed args
VAL_RATIO="0.15"
TEST_RATIO="0.2"
N_RUNS="3"
# MEM_MODE="unlim_mem"
W_MODE="fixed"

# DGB datasets (https://zenodo.org/records/7213796#.Y1cO6y8r30o)
DATA_ARGS=("wiki" "uci" "mooc" "USLegis" "SocialEvo" "UNvote") # "wiki" "uci" "mooc" 

# Negative sampling modes
SAMPLING_MODES=("rnd" "induc_nre" "hist_nre")

MEM_MODES=("unlim_mem" "time_window")

for DATA in "${DATA_ARGS[@]}"
do
    for NEG_SAMPLE in "${SAMPLING_MODES[@]}"
    do
        for MEM_MODE in "${MEM_MODES[@]}"
        do 
            echo "Running on dataset: $DATA | Sampling: $NEG_SAMPLE"
            $PYTHON baseline.py \
                --data "$DATA" \
                --val_ratio "$VAL_RATIO" \
                --test_ratio "$TEST_RATIO" \
                --n_runs "$N_RUNS" \
                --neg_sample "$NEG_SAMPLE" \
                --mem_mode "$MEM_MODE" \
                --w_mode "$W_MODE"
            echo "-------------------------------------------------------------"
        done
    done
done