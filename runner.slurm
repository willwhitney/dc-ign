#!/bin/bash
#SBATCH --job-name=cache
#SBATCH --output=slurm_cache.out
#SBATCH --error=slurm_cache.err


th cache.lua
#OMP_NUM_THREADS=1 th main.lua -t 1 -s logs_150rgb
