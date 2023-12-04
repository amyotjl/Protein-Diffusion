import torch
from torch.utils.data import Dataset, DataLoader
from torch import Tensor, nn
from denoising_diffusion_pytorch import Unet1D, GaussianDiffusion1D, Trainer1D
import joblib
from utils import compute_stats, fid
import json
import glob
import re
import argparse
import os
import numpy as np
import sklearn
from utils import reconstruct_sequence
import sklearn.metrics

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def compute_pairwise_distance(data_x, data_y=None):
    """
    Args:
        data_x: numpy.ndarray([N, feature_dim], dtype=np.float32)
        data_y: numpy.ndarray([N, feature_dim], dtype=np.float32)
    Returns:
        numpy.ndarray([N, N], dtype=np.float32) of pairwise distances.
    """
    if data_y is None:
        data_y = data_x
    dists = sklearn.metrics.pairwise_distances(
        data_x, data_y, metric='euclidean', n_jobs=8)
    return dists


def get_kth_value(unsorted, k, axis=-1):
    """
    Args:
        unsorted: numpy.ndarray of any dimensionality.
        k: int
    Returns:
        kth values along the designated axis.
    """
    indices = np.argpartition(unsorted, k, axis=axis)[..., :k]
    k_smallests = np.take_along_axis(unsorted, indices, axis=axis)
    kth_values = k_smallests.max(axis=axis)
    return kth_values


def compute_nearest_neighbour_distances(input_features, nearest_k):
    """
    Args:
        input_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        nearest_k: int
    Returns:
        Distances to kth nearest neighbours.
    """
    distances = compute_pairwise_distance(input_features)
    radii = get_kth_value(distances, k=nearest_k + 1, axis=-1)
    return radii


def compute_prdc(real_features, fake_features, nearest_k):
    """
    Computes precision, recall, density, and coverage given two manifolds.

    Args:
        real_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        fake_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        nearest_k: int.
    Returns:
        dict of precision, recall, density, and coverage.
    """

    print('Num real: {} Num fake: {}'
          .format(real_features.shape[0], fake_features.shape[0]))

    real_nearest_neighbour_distances = compute_nearest_neighbour_distances(
        real_features, nearest_k)
    fake_nearest_neighbour_distances = compute_nearest_neighbour_distances(
        fake_features, nearest_k)
    distance_real_fake = compute_pairwise_distance(
        real_features, fake_features)

    precision = (
            distance_real_fake <
            np.expand_dims(real_nearest_neighbour_distances, axis=1)
    ).any(axis=0).mean()

    recall = (
            distance_real_fake <
            np.expand_dims(fake_nearest_neighbour_distances, axis=0)
    ).any(axis=1).mean()

    density = (1. / float(nearest_k)) * (
            distance_real_fake <
            np.expand_dims(real_nearest_neighbour_distances, axis=1)
    ).sum(axis=0).mean()

    coverage = (
            distance_real_fake.min(axis=1) <
            real_nearest_neighbour_distances
    ).mean()

    return dict(precision=precision, recall=recall,
                density=density, coverage=coverage)


#loading the model
def evaluate_folder(folder_name, dataset, number_seq_generate, autoencoder, random_orth_matrix):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        checkpoint = match.group(1)
        #Loading the checkpoint
        trainer.load(checkpoint)

        #Generating the sequences
        samples = diffusion.sample(number_seq_generate)

        prdc = compute_prdc(dataset['inputs'].squeeze().detach().cpu().numpy(), samples.detach().squeeze().cpu(), 5)

        #Decode the sequences
        decoder = autoencoder.decoder.to(device)
        decoded = decoder(samples)
        new_sequences = [''.join(x) for x in reconstruct_sequence(random_orth_matrix, decoded.detach().cpu())]

        with open(os.path.join(folder_name, 'generated_sequences_{}.txt'.format(checkpoint)), 'w') as f:
            for seq in new_sequences:
                f.write(seq + '\n')

        #Computing stats for generated samples
        mu_samples, sig_samples = compute_stats(samples.squeeze().detach().cpu().numpy())

        #Computing FID
        fids.append((checkpoint,fid(mu_data, mu_samples, sig_data, sig_samples),prdc))
        fids.sort(key=lambda x: x[0])
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
    output_file = 'result1.json'

    autoencoder = joblib.load(args.autoencoder)
    random_orth_matrix = joblib.load(args.rom)

    # Loop over all subfolders
    for folder in sorted_folders:
        if os.path.isdir('{}/{}'.format(diffusion_folder, folder)):

            #If evaluation hasn't started for this folder, we go in and "lock" it by creating an empty result file
            if output_file not in os.listdir('{}/{}'.format(diffusion_folder, folder)):
                with open(os.path.join(diffusion_folder, folder, output_file), 'w') as f:
                    json.dump({}, f, indent=4)

                print('Evaluating {}'.format(folder))

                #Launch the evaluation
                fids= evaluate_folder('{}/{}'.format(diffusion_folder, folder), data, number_seq_generated, autoencoder, random_orth_matrix)
                print(fids, folder)

                results[folder] = {'fids': fids}

                #Write fids to result file
                with open(os.path.join(diffusion_folder, folder, output_file), 'w') as f:
                    json.dump(results[folder], f, indent=4)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-data', type=str, help="Path to the Joblib file containing the LATENT data formated in \{inputs, sequences\} (The result of script create_orth_dataset.py)", required=True)
    parser.add_argument('-f', '--folder', type=str, help='Root folder where all model where saved', required=True)
    parser.add_argument('-g', '--generate_seq', help='Number of sequences to generate to compute the FID', default=2048, type=int)
    parser.add_argument('-rom', type=str, help='path to the random orthogonal matrix stored in a JOBLIB file', required=True)
    parser.add_argument('-a', '--autoencoder', type=str, help='File where the autoencoder model is saved ', required=True)
    args = parser.parse_args()

    evaluate(args)
