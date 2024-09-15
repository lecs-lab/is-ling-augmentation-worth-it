#!/bin/bash
#SBATCH --nodes=1           # Number of requested nodes
#SBATCH --gres=gpu:2
#SBATCH --ntasks=2         # Number of requested cores
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00          # Max walltime              # Specify QOS
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

for size in 10 100 500 1000 5000
do
    for model in baseline aug_m1 aug_m2
    do
        torchrun --nproc_per_node=2 mt_experiments.py train \
                                        --model_type $model \
                                        --aug_mode mixed \
                                        --direction "usp->esp" \
                                        --sample_train_size $size
    done
    # Run without a train sample size, ie all data
    torchrun --nproc_per_node=2 mt_experiments.py train \
                                    --model_type baseline \
                                    --aug_mode mixed \
                                    --direction "usp->esp"
done
