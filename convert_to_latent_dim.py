import torch
import joblib
from matplotlib import pyplot as plt
import os
import gc
from load_dataset import get_input_seqs_dataloader
import utils
import shutil
from Architectures.decoder_conv import Decoder
from Architectures.encoder_conv import Encoder
from Architectures.autoencoder_conv_pl import Autoencoder
import argparse
from torch.utils.data import DataLoader
import torch


def encode_data(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = joblib.load(args.data)
    model = joblib.load(args.model).to(device)
    loader = DataLoader(list(zip(data['inputs'], data['sequences'])), batch_size=64)
    outputs = []
    seqs = []
    for batch in loader:
        data, seq = batch
        data=data.to(device)
        output = model.encoder(data)
        outputs.append(output)
        seqs.append(seq)

    joblib.dump({"inputs":torch.cat(outputs).unsqueeze(1).detach().cpu(), "sequences":seqs}, args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--model', type=str, help='Path to the Joblib file containing the autoencoder model', required=True)
    parser.add_argument('-data', type=str, help="Path to the Joblib file containing the data formated in \{inputs, sequences\} (The result of script create_orth_dataset.py)", required=True)
    parser.add_argument('-out', '--output_dir', type=str, help='Path where to save the encoded dataset', required=True)

    args = parser.parse_args()
    encode_data(args)