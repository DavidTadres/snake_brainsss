#!/bin/bash
#SBATCH --job-name=stitch_nii
#SBATCH --partition=trc
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=./logs/stitchlog3.out
#SBATCH --open-mode=append
#SBATCH --mail-type=ALL

ml python/3.9.0
source /home/users/dtadres/.env_snakemake/bin/activate
date
snakemake
