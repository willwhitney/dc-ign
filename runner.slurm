#!/bin/bash
#SBATCH --job-name=visualize_networks
#SBATCH --output=visualize_networks.out
#SBATCH --error=visualize_networks.err

th visualize_networks.lua
