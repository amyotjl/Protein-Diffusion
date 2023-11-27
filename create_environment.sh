#!/bin/bash
# load the miniconda module
module load miniconda3-4.8.2-gcc-9.2.0-sbqd2xu
# create a conda environment with python 3.7 named pytorch
conda create --name $1 python=3.7
source activate $1
# install pytorch dependencies via conda
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install scipy -c conda-forge
pip install denoising_diffusion_pytorch 