#!/bin/bash
#SBATCH --job-name=F32_H2
#SBATCH --output=F32_H2.out
#SBATCH --error=F32_H2.err

##th cache.lua
th main.lua -t1 -s "F32_H2"
