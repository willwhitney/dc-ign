#!/bin/bash


# #SBATCH --job-name=crashtest
# #SBATCH --output=crashtest.out
# #SBATCH --error=crashtest.err

# th crashtest.lua



#SBATCH --job-name=movie_maker
#SBATCH --output=movie_maker.out
#SBATCH --error=movie_maker.err

th movie_maker.lua



# #SBATCH --job-name=visualize_networks
# #SBATCH --output=visualize_networks.out
# #SBATCH --error=visualize_networks.err

# th visualize_networks.lua
