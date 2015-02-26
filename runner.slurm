#!/bin/bash
#SBATCH --job-name=test1
#SBATCH --output=test1.out
#SBATCH --error=test1.err

th monovariant_main.lua --save 'test-thing' --import 'F96_H120_lr0_0005' --shape_bias --shape_bias_amount 10
