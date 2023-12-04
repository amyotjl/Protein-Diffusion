import torch
from denoising_diffusion_pytorch import Unet1D, GaussianDiffusion1D, Trainer1D, Unet, Trainer, GaussianDiffusion
import joblib

import glob
import os
import re
from utils import reconstruct_sequence
import json
import argparse

def load_diffusion_model(folder):
    with open('{}/config.json'.format(folder), 'r') as f:
        config = json.load(f)

    model = Unet1D(
        **config['unet'],
    ).cuda()

    diffusion = GaussianDiffusion1D(
        model,
        **config['gaussian'],
    ).cuda()

    trainer = Trainer1D(
        diffusion,
        dataset = torch.Tensor(101,1),
        results_folder=folder,
        **config['trainer']
    )

    list_of_files = glob.glob('{}/*.pt'.format(folder)) # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)
    pattern = r"-(.*?)\."
    match = re.search(pattern, latest_file)
    if match:
        result = match.group(1)
        trainer.load(result)

    return model, diffusion, trainer

def generate(args):

    autoencoder = joblib.load(args.autoencoder)
    rand_orth_mat = joblib.load(args.rom)


    #Loading the diffusion model
    model, diffusion, trainer = load_diffusion_model(args.diffusion_folder)

    # Generating new latent vector
    samples = []
    for _ in range(args.generate_seq//args.batch_size):
        samples.append(diffusion.sample(batch_size=args.batch_size))

    samples = torch.cat(samples).detach().cpu()

    # Decoding the latent vector
    decoded = autoencoder.decoder(samples)

    new_sequences = [''.join(x) for x in reconstruct_sequence(rand_orth_mat, decoded.detach().cpu())]

    with open(args.output_file, 'w') as f:
        for seq in new_sequences:
            f.write(seq + '\n')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--diffusion_folder', type=str, help='Folder where the diffusion model is saved with the config file', required=True)
    parser.add_argument('-rom', type=str, help='path to the random orthogonal matrix stored in a JOBLIB file', required=True)
    parser.add_argument('-a', '--autoencoder', type=str, help='File where the autoencoder model is saved ', required=True)
    parser.add_argument('-g', '--generate_seq', type=int, help='Number of sequences to generate to compute the FID', default=2048)
    parser.add_argument('-bs', '--batch_size', type=int, help='Batch size')
    parser.add_argument('-out', '--output_file', type=str, help='Name of the JSON file to write the output all results ', default='new_sequences.txt')
    args = parser.parse_args()

    generate(args)
