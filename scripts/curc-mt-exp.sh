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

if [[ "$1" != "baseline" && "$1" != "aug_m1" && "$1" != "aug_m2" && "$1" != "combo" ]]; then
    echo "Error: First argument must be 'baseline', 'aug_m1', or 'aug_m2'."
    exit 1
fi

model=$1
shift


module purge
module load gcc/11.2.0
source /curc/sw/anaconda3/latest
conda activate AutoIGT

export STANZA_RESOURCES_DIR="/scratch/alpine/migi8081/stanza/"

cd "/projects/migi8081/augmorph/src"

for size in 50 100 300 500 1000 5000
do
    for seed in 0 1 2
    do
        python mt_experiments.py train \
                                    --augmentation_type $model \
                                    --direction "usp->esp" \
                                    --sample_train_size $size \
                                    --seed $seed
    done
done

for seed in 0 1 2
do
    # Run without a train sample size, ie all data
    python mt_experiments.py train \
                                --augmentation_type $model \
                                --direction "usp->esp" \
                                --seed $seed
done
