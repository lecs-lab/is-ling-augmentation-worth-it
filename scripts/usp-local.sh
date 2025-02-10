#!/bin/bash

# Generate all possible combinations of flags
AUG_FLAGS=(--run-random-insert-conj --run-tam-update --run-random-duplicate --run-random-delete --run-delete-w-exclusions --run-random-insert-noise)
NUM_FLAGS=${#AUG_FLAGS[@]}
TOTAL_COMBOS=$((1 << NUM_FLAGS))

for ((i=0; i<TOTAL_COMBOS; i++)); do
    ARGS=()
    for ((j=0; j<NUM_FLAGS; j++)); do
        # Check if the j-th bit in i is set
        if (( i & (1 << j) )); then
            ARGS+=("${AUG_FLAGS[j]}")
        fi
    done

    >&2 echo "RUNNING EXPERIMENT WITH AUG FLAGS: ${ARGS[@]}"

    for direction in "transc->transl" "transl->transc" "transc->gloss"
    do
        for size in 100 500 1000 5000
        do
            for seed in 0 1 2
            do
                >&2 echo "DIRECTION: $direction - SIZE: $size - SEED: $seed"
                python src/train.py --language usp \
                                    --direction $direction \
                                    --sample_train_size $size \
                                    --seed $seed \
                                    "${ARGS[@]}"
            done
        done

        for seed in 0 1 2
        do
            >&2 echo "DIRECTION: $direction - SIZE: full - SEED: $seed"
            # Run without a train sample size, ie all data
            python src/train.py --direction $direction \
                                --seed $seed \
                                "${ARGS[@]}"
        done
    done
done
