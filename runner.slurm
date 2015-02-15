#!/bin/bash
#SBATCH --job-name=log150g001_R2
#SBATCH --output=log150g001_R2.out
#SBATCH --error=log150g001_R2.err

#th cache.lua
th main.lua -t1 -u1 --network log150g001/vxnet.net  -slog150g001_R2
