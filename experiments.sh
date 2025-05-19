#!/bin/bash

# Batch script for running Base3 with various settings

# Fixed/default args
BS="200"
SEED="2"
MEM_MODE="time_window"
W_MODE="fixed"
METHOD="multi_conf"
CO_OCCUR_WEIGHT="1.0"

# Hyperparameters to sweep
MEM_SPANS=("0.0001" "0.001" "0.01" "0.1" "1.0")
K_VALUES=("50" "100" "500" "1000" "50000")
SAMPLING_MODES=("rnd" "induc_nre" "hist_nre" "rp_ns")
DATASETS=("tgbl-wiki")  

for DATA in "${DATASETS[@]}"
do
    for NEG_SAMPLE in "${SAMPLING_MODES[@]}"
    do
        for MEM_SPAN in "${MEM_SPANS[@]}"
        do
            for K in "${K_VALUES[@]}"
            do
                echo "Running: DATA=$DATA | NS=$NEG_SAMPLE | K=$K | MS=$MEM_SPAN"

                python baseline.py \
                    --data "$DATA" \
                    --bs "$BS" \
                    --seed "$SEED" \
                    --neg_sample "$NEG_SAMPLE" \
                    --mem_mode "$MEM_MODE" \
                    --w_mode "$W_MODE" \
                    --mem_span "$MEM_SPAN" \
                    --k_value "$K" \
                    --co_occurrence_weight "$CO_OCCUR_WEIGHT" \
                    --method "$METHOD"

                echo "-------------------------------------------------------------"
            done
        done
    done
done