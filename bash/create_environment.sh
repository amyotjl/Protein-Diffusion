#!/bin/bash
# load the miniconda module
module load miniconda3-4.8.2-gcc-9.2.0-sbqd2xu

conda env create -f environment.yml
source activate diffusion

pip install ema-pytorch
pip install denoising_diffusion_pytorch pytorch-fid --no-deps
conda install einops -c conda-forge
conda install matplotlib -c conda-forge
