#!/bin/bash
# load the miniconda module
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16

module load miniconda3-4.8.2-gcc-9.2.0-sbqd2xu

source activate plz

srun python ../generate_sequences.py -d $1 -rom $2 -a $3 -g $5 -bs $6 -out $7
