#!/bin/bash

#SBATCH --begin=now
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --mail-user=naureen.ghani.18@ucl.ac.uk 
#SBATCH --output=my_script.o%j

set -e

# Re-submit job script for the next execution
sbatch splitChannelsGPU.sbatch

# Do planned work