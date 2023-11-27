import torch
from torch.utils.data import Dataset, DataLoader
from torch import Tensor, nn
from denoising_diffusion_1D import Unet1D, GaussianDiffusion1D, Trainer1D
import joblib
from utils import compute_stats, fid
import json
import glob
import re
import argparse
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#loading the model
def evaluate_folder(folder_name, dataset, number_seq_generate):
    with open('{}/config.json'.format(folder_name), 'r') as f:
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
        dataset = torch.Tensor(101,1), #not important since we don't train
        results_folder=folder_name,
        **config['trainer']
    )

    # Loading the decoder
    fids = []
    list_of_files = glob.glob('{}/*.pt'.format(folder_name)) # * means all if need specific format then *.csv
    
    #Compute stats for real data. Needs to be computed once only. 
    mu_data, sig_data = compute_stats(dataset['inputs'].squeeze().detach().cpu().numpy())

    # We generate samples for each checkpoints (there might be only one in the folder)
    for file in list_of_files:
        pattern = r"-(.*?)\." #gets the checkpoint number to load 
        match = re.search(pattern, file)

        #Loading the checkpoint
        trainer.load(match.group(1)) 

        #Generating the sequences
        samples = diffusion.sample(number_seq_generate)

        #Computing stats for generated samples
        mu_samples, sig_samples = compute_stats(samples.squeeze().detach().cpu().numpy())

        #Computing FID
        fids.append(fid(mu_data, mu_samples, sig_data, sig_samples))

    return fids

def evaluate(args):
    diffusion_folder = args.folder
    results = {}
    # List all folders in the given directory
    folders = [folder for folder in os.listdir(diffusion_folder) if os.path.isdir(os.path.join(diffusion_folder, folder))]

    data = joblib.load(args.data)
    number_seq_generated = args.generate_seq

    # Sort the folders based on the latest modification time
    sorted_folders = sorted(folders, key=lambda folder: os.path.getmtime(os.path.join(diffusion_folder, folder)), reverse=False)
    output_file = '{}.json'.format(args.output_file)

    # Loop over all subfolders
    for folder in sorted_folders:
        if os.path.isdir('{}/{}'.format(diffusion_folder, folder)):

            #If evaluation hasn't started for this folder, we go in and "lock" it by creating an empty result file
            if output_file not in os.listdir('{}/{}'.format(diffusion_folder, folder)):
                with open(os.path.join(diffusion_folder, folder, output_file), 'w') as f:
                    json.dump({}, f, indent=4)

                print('Evaluating {}'.format(folder))

                #Launch the evaluation
                fids= evaluate_folder('{}/{}'.format(diffusion_folder, folder), data, number_seq_generated)
                print(fids, folder)

                results[folder] = {'fids': fids}

                #Write fids to result file
                with open(os.path.join(diffusion_folder, folder, output_file), 'w') as f:
                    json.dump(results[folder], f, indent=4)


if __name__ == "__main__":    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', type=str, help="Path to the Joblib file containing the LATENT data formated in \{inputs, sequences\} (The result of script create_orth_dataset.py)", required=True)
    parser.add_argument('-f', '--folder', type=str, help='Root folder where all model where saved', required=True)
    parser.add_argument('-out', '--output_file', type=str, help='Name of the JSON file to write the output (FID). ', default='result')
    parser.add_argument('-g', '--generate_seq', help='Number of sequences to generate to compute the FID', default=2048, type=int)
    args = parser.parse_args()

    evaluate(args)