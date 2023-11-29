#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32


module load miniconda3-4.8.2-gcc-9.2.0-sbqd2xu

source activate plz

srun python ../convert_to_latent_dim.py -m $1 -data  $2 -out $3