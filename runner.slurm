#!/bin/bash
#SBATCH --job-name=TORCH7_DEEPSHIT
#SBATCH --output=slurm.out
#SBATCH --error=slurm.err

th cache.lua
