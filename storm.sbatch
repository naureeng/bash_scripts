#!/bin/bash 
# 
#SBATCH -p gpu 
# partition (queue) #SBATCH -N 1 #number of nodes 
#SBATCH -n 1 # number of cores 
##SBATCH --gres gpu:rtx2080:1
##SBATCH -w gpu-380-13
#SBATCH --gres gpu:1
#SBATCH --mem=200000
#SBATCH --mail-user=<naureen.ghani.18@ucl.ac.uk>

module load miniconda
conda activate storm
python test.py 
