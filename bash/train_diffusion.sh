#!/bin/bash
# load the miniconda module
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16

module load miniconda3-4.8.2-gcc-9.2.0-sbqd2xu

source activate diffusion

#Example
#sbatch train_diffusion.sh "../data/latent_dim_28_180.joblib" "../testDiffusion" 16

srun python ../train_diffusion1D.py -data $1 -out $2 -d $3
