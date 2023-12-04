#!/bin/bash
# load the miniconda module
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16

module load miniconda3-4.8.2-gcc-9.2.0-sbqd2xu

source activate diffusion

#Example
# sbatch generate_sequences.sh "../Diffusion" "../data/AA_random_matrices_orth.joblib" "../Autoencoder/autoencoder" 2048 512

srun python ../generate_sequences.py -d $1 -rom $2 -a $3 -g $4 -bs $5 
