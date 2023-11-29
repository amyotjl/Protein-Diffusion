#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32


module load miniconda3-4.8.2-gcc-9.2.0-sbqd2xu

source activate plz

srun python ../merge_autoencoder_evaluation.py -f $1 -res None