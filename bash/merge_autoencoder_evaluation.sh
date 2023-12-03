#!/bin/bash

#SBATCH --cpus-per-task=8


module load miniconda3-4.8.2-gcc-9.2.0-sbqd2xu

source activate diffusion

#Example
# sbatch merge_autoencoder_evaluation.sh "../testAutoencoder" "../testAutoencoder/results.txt"
srun python ../merge_autoencoder_evaluation.py -f $1 -res $2
