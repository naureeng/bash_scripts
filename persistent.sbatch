#!/bin/bash
#SBATCH --job-name=persistent
#SBATCH --dependency=singleton
#SBATCH --time=01:00:00
#SBATCH --signal=B:SIGUSR1@90
#SBATCH -p gpu 
# partition (queue) #SBATCH -N 1 #number of nodes 
#SBATCH -n 1 # number of cores 
##SBATCH --gres gpu:rtx2080:1
##SBATCH -w gpu-380-13
#SBATCH --gres gpu:1
#SBATCH --mem=200000


# catch the SIGUSR1 signal
_requeue() {
    sbatch splitChannelsGPU.sbatch
    scontrol requeue $SLURM_JOBID
}
trap '_requeue' SIGUSR1

## Insert the command to run below. Here, we're just outputing the date every
## 60 seconds, forever

while true; do
    sbatch splitChannelsGPU.sbatch
    sleep 3600
done
