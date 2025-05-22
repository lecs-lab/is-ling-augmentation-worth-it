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
conda activate igt
export STANZA_RESOURCES_DIR="/scratch/alpine/migi8081/stanza/"
cd "/projects/migi8081/augmorph/"

for size in 100 500 1000 5000
do
    for seed in 0 1 2
    do
        python src/train.py --language arp \
                            --direction "transl->transc" \
                            --sample_train_size $size \
                            --seed $seed
        python src/train.py --language arp \
                            --direction "transl->transc" \
                            --sample_train_size $size \
                            --seed $seed \
                            --run-sentence-permutations
    done
done
