#!/bin/bash
# load the miniconda module
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16

module load miniconda3-4.8.2-gcc-9.2.0-sbqd2xu

source activate diffusion

#Example
# sbatch evaluate_diffusion1d.sh "../data/latent_test" "../testDiffusion"  2048

srun python ../evaluate_diffusion1d.py -data $1 -f $2  -g $3
