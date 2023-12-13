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

## 3. Autoencoder

### 3.1 Training using grid search

*Architechtures/* contains the code for the autoencoder. I used [Pytorch Lightning](https://lightning.ai/docs/pytorch/stable/) to design it.

To train the autoencoders and perform gridsearch, use the train_autoencoder.sh script. For example: 
```
sbatch train_autoencoder.sh "../data/orth_dataset" "../data/AA_random_matrices_orth.joblib" "../testAutoencoder" 5 64 64
```

This will output a single file per hyperparameters combinasion containing the accuracy for all tested folds and the average. 

### 3.2 Merging results

Since each models are outputting a single file, we want to merge all the results into one file to ge the best models. The script *merge_autoencoder_evalutation.sh* does that for us. Use the command: 

```
sbatch merge_autoencoder_evaluation.sh "../testAutoencoder" "../testAutoencoder/results.txt"
```

You can now manually check the best autoencoder and its accuracy.

## 3.3 Training best model

Since cross validation is used and it does not use the whole dataset for training, we train a final model using the best hyperparameters on the whole dataset. This final model will be use to encode the data and decode the diffusion model's output.

TODO - Write script

## 4. Encoding 

Once we have our final autoencoder, we need to encode the dataset to its latent representation. For this, we run the script *convert_to_latent_dim.sh* like such : 

```
sbatch convert_to_latent_dim.sh ../Autoencoder/autoencoder "../data/orth_dataset" "../data/latent_test"
```

Use should now have a data file usable to train the diffusion model.

## 5. Diffusion
### 5.1 Training grid search

Using the file from step 4, we can train diffusion models. We use the library  [Denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch) to do so. We are using the 1D architecture. The script *train_diffusion.sh* does the trick.

```
sbatch train_diffusion.sh "../data/latent_dim_28_180.joblib" "../testDiffusion" 16
```

Each combination of hyperparameters will create a folder. Inside that folder, a configuration file will be created. It is useful to load a model or checkpoint. Inside that folder, checkpoints will be saved. Default configuration is 10 000 training steps and 2 000 steps between checkpoints.

Many of this script may be launch at a few seconds of interval.

### 5.2 Evalutation

Each folder will be evaluated for each of their checkpoints. Evaluation script generates fake sample and computes the FID, density and coverage against the original dataset. This takes time. Run the following command (with possibly different parameters).

```
sbatch evaluate_diffusion1d.sh "../data/latent_test" "../testDiffusion"  16 "../Autoencoder/autoencoder" "../data/AA_random_matrices_orth.joblib"
```

This outputs a result file in the evaluated directory. Initially, the file will be blank as it serves as a "lock" to prevent testing the same directory on multiple processes.

Run many evalutation script to speed up the process. 

### 5.3 Merging evaluation

To merge all evaluation from all diffusion directories, run the script *merge_diffusion_evaluation.sh*. For example: 

```
sbatch merge_diffusion_evaluation.sh "../testDiffusion"
```

This is the final step. Once yo analyze the results, the models are already savec in their respective folder. 

The folder *Diffusion* holds the best performing model based on FID.