#!/bin/bash
# load the miniconda module
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16

module load miniconda3-4.8.2-gcc-9.2.0-sbqd2xu

source activate plz

srun python ../evaluate_diffusion1d.py -data $1 -f $2 -d $3 -g $4
