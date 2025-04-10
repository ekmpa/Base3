#!/bin/bash

# Fixed args
VAL_RATIO="0.15"
TEST_RATIO="0.15"
N_RUNS="1"
MEM_MODE="time_window"
W_MODE="fixed"

W_MODES=("fixed" "avg_reoccur")

MEM_SPANS=("0.0001" "0.001" "0.01" "0.1" "1")

# DGB datasets (https://zenodo.org/records/7213796#.Y1cO6y8r30o)
DATA_ARGS=("tgbl-review") # "tgbl-flight" "tgbl-coin" "tgbl-review" "tgbl-comment" 
#DATA_ARGS=("enron" "uci" "mooc" "USLegis" "UNvote" "CanParl" "SocialEvo") # "wiki" "uci" "uci"  

# Negative sampling modes
SAMPLING_MODES=("rnd" "induc_nre" "hist_nre" "rp_ns") # "rnd"

# K for Poptrack
K_VALS=("50" "100" "500" "1000" "50000")

#MEM_MODES=("unlim_mem" "time_window")

for DATA in "${DATA_ARGS[@]}"
do
    for NEG_SAMPLE in "${SAMPLING_MODES[@]}"
    do
        for K in "${K_VALS[@]}"
        do 
            echo "Running on dataset: $DATA | Sampling: $NEG_SAMPLE"
            python baseline.py \
                --data "$DATA" \
                --val_ratio "$VAL_RATIO" \
                --test_ratio "$TEST_RATIO" \
                --n_runs "$N_RUNS" \
                --neg_sample "$NEG_SAMPLE" \
                --mem_mode "$MEM_MODE" \
                --w_mode "$W_MODE" \
                --k_val "$K" 

            echo "-------------------------------------------------------------"
        done
    done
done