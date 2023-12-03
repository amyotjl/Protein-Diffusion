#!/bin/bash
# load the miniconda module
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16

module load miniconda3-4.8.2-gcc-9.2.0-sbqd2xu

source activate diffusion

#Example
#sbatch train_autoencoder.sh "../data/orth_dataset" "../data/AA_random_matrices_orth.joblib" "../testAutoencoder" 5 64 64

srun python ../train_autoencoder1D_gridSearch.py -data $1 -rom $2 -out $3 -cv $4 -bs $5 -ld $6
