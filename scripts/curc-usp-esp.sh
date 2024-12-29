#!/bin/bash
#SBATCH --nodes=1           # Number of requested nodes
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1         # Number of requested cores
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00          # Max walltime              # Specify QOS
#SBATCH --qos=blanca-curc-gpu
#SBATCH --partition=blanca-curc-gpu
#SBATCH --account=blanca-curc-gpu
#SBATCH --out=log.%j.out      # Output file name
#SBATCH --error=log.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=michael.ginn@colorado.edu

module purge
module load gcc/11.2.0
source /curc/sw/anaconda3/latest
conda activate AutoIGT
export STANZA_RESOURCES_DIR="/scratch/alpine/migi8081/stanza/"
cd "/projects/migi8081/augmorph/"

# cd ../src

# Generate all possible combinations of flags
AUG_FLAGS=(--run-random-insert-conj --run-tam-update --run-random-duplicate --run-random-delete --run-delete-w-exclusions)
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

    for size in 50 100 300 500 1000 5000
    do
        for seed in 0 1 2
        do
            python src/train.py --direction "transc->transl" \
                                --sample_train_size $size \
                                --seed $seed \
                                "${ARGS[@]}"
        done
    done

    for seed in 0 1 2
    do
        # Run without a train sample size, ie all data
        python src/train.py --direction "transc->transl" \
                            --seed $seed \
                            "${ARGS[@]}"
    done
done
