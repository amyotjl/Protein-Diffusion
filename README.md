# Protein-Diffusion

This repository contains the files used during the project. Pretrained models can be found here:

* Autoencoder &rarr; Autoencoder/autoencoder
* Diffusion &rarr; Diffusion/model-1.pt

Here are the steps and files to reproduce the project. All command shown are for Trixie (SLURM commands) :

## 1. Create conda environment

To create the environment and the required libraries, run the following command from starting at this root directory:

```
cd bash
sbatch create_environment.sh
```

A new conda environment will be created. Its name is *diffusion*.

## 2. Data preprocessing

Many data files can be found in *data/* . The *Original* folder contains the original data, but it contains duplicates. They are not relevant anymore, but left there in case of expension or anything else to improve the results or project.

You need to create the vectorized dataset to go from sequences to matrices for inputs if you ever want to re-train something. The file was too big for Github, so you'll need to generate it yourself. 

```
sbatch create_orth_dataset.sh "../data/AA_28_180.fasta" "../data/AA_random_matrices_orth.joblib" "../data/orth_dataset" 28 180
```

*../data/orth_dataset*  can be changed to any location/name.

Data files description:

* AA_28_180.fasta &rarr; Sequences of length between 28 and 180
* AA_random_matrices_orth.joblib &rarr; Matrix of random orthogonal vectors. There are 21 unique amino acid and 1 padding character. So the dimension of this matrix is 22x22.
* laten_dim_28_180.joblib &rarr; Encoded dataset to the latent space using the Encoder's component of the autoencoder. 

## 3. Autoencoder training

*Architechtures/* contains the code for the autoencoder. I used Pytorch Lightning to design it.   


