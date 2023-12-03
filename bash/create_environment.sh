#!/bin/bash
# load the miniconda module
module load miniconda3-4.8.2-gcc-9.2.0-sbqd2xu
# create a conda environment with python 3.7 named pytorch
conda env create -f environment.yml
source activate diffusion
# install pytorch dependencies via conda
# conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
# conda install scipy joblib scikit-learn -c conda-forge
pip install ema-pytorch
pip install denoising_diffusion_pytorch pytorch-fid --no-deps
conda install einops -c conda-forge
conda install matplotlib -c conda-forge
