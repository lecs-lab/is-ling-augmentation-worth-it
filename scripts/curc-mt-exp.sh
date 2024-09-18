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
source /curc/sw/anaconda3/latest

conda activate AutoIGT

export STANZA_RESOURCES_DIR="/scratch/alpine/migi8081/stanza/"

cd "/projects/migi8081/morpheme-hallucination/src"

if [[ "$1" != "baseline" && "$1" != "aug_m1" && "$1" != "aug_m2" ]]; then
    echo "Error: First argument must be 'baseline', 'aug_m1', or 'aug_m2'."
    exit 1
fi

model=$1

for size in 300 500 800 1000 5000
do
    python mt_experiments.py train \
                                --model_type $model \
                                --aug_mode mixed \
                                --direction "esp->usp" \
                                --sample_train_size $size
done

# Run without a train sample size, ie all data
python mt_experiments.py train \
                            --model_type $model \
                            --aug_mode mixed \
                            --direction "esp->usp"
