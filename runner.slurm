#!/bin/bash
#SBATCH --job-name=log150g001F96_R2
#SBATCH --output=log150g001F96_R2.out
#SBATCH --error=log150g001F96_R2.err

##th cache.lua
th main.lua -t1 -u1 -n"log150gl001F96/params.t7"  -slog150gl001F96_R2
