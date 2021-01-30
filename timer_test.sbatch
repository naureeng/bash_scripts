#!/bin/bash

#SBATCH --job-name=timertest
#SBATCH --output=timer_test.out
#SBATCH --error=timer_test.err
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=16

module load matlab/R2018a

matlab -nodisplay < timer_test.m 