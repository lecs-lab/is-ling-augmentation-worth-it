#!/bin/bash
#SBATCH --nodes=1           # Number of requested nodes
#SBATCH --gres=gpu:1
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

# Check args
case "$1" in
    igt|mt|segment)
        echo "Running mode: $1"
        ;;
    *)
        echo "Error: Invalid first argument. Must be 'igt', 'mt', or 'segment'"
        exit 1
        ;;
esac
MODE=$1
shift


module purge
source /curc/sw/anaconda3/latest

conda activate AutoIGT

cd "/projects/migi8081/morpheme-hallucination/src"

if [ "$MODE" == "igt" ]; then
    torchrun --nproc_per_node=1 igt_experiments.py train --model_type baseline --aug_mode mixed "$@"
elif [ "$MODE" == "mt" ]; then
    python mt_experiments.py train --model_type baseline --aug_mode mixed --direction "usp->esp"
    python mt_experiments.py train --model_type aug_m1 --aug_mode mixed --direction "usp->esp"
    python mt_experiments.py train --model_type aug_m2 --aug_mode mixed --direction "usp->esp"
elif [ "$MODE" == "segment" ]; then
    echo "Error: segmentation not yet implemented"
    exit 1
    torchrun --nproc_per_node=1 segment_experiments.py train --model_type baseline --aug_mode mixed "$@"
fi
