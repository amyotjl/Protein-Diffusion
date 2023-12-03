#!/bin/bash
# load the miniconda module
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16

module load miniconda3-4.8.2-gcc-9.2.0-sbqd2xu

source activate diffusion

#Example of a command to run this bash file
#sbatch create_orth_dataset.sh "../data/AA_28_180.fasta" "../data/AA_random_matrices_orth.joblib" "../data/orth_dataset" 28 180


srun python ../create_orth_dataset.py -fp $1 -rom $2 -out $3 -min $4 -max $5
