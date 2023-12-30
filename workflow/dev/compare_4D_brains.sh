#!/bin/bash
#SBATCH --job-name=4D_array_comparison
#SBATCH --partition=trc
#SBATCH --time=4-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --output=./logs/mainlog.out
#SBATCH --open-mode=append
#SBATCH --mail-type=ALL

ml python/3.9.0
source .env_snakemake/bin/activate
python3 compare_4D_brains.py