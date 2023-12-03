import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import os
import gc
import utils
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from Architectures.autoencoder_conv_pl import Autoencoder
from pathlib import Path
import argparse


def run(args):
    data = joblib.load(args.data)
    data['sequences'] = np.array(data['sequences'])
    random_matrices = joblib.load(args.rom_path)

    Path(args.output_dir).mkdir(exist_ok=True)
    config = {
        'output_shape': [(180,22)],
        'epochs': [30],
        'bs':[args.batch_size],
        'latent_dim':[args.latent_dim],
        'channels':[[180,512,256],[180,512,64], [180, 512, 256, 128], [180,512,256,128,64]],
        'conv_k': list(range(1,4)),
        'conv_s': list(range(1,3)),
        'conv_p': [0],
        'activation_func': [nn.Sigmoid, nn.ReLU, nn.Tanh],
        'loss_func': [nn.MSELoss(), nn.L1Loss()],
        'optimizer' : [optim.SGD, optim.Adagrad, optim.AdamW],
        'dropout_rate' : [0],
        'batch_normalization' : [False]
    }

    kf = KFold(args.cross_validation)
    used_model = []
    if args.result_file: 
        with open(args.result_file, 'r') as f:
            parsed = f.read()
            used_model = parsed.split('\n')
            f.close()



    torch.set_float32_matmul_precision('medium')
    for conf in utils.get_combinations(config):
            gc.collect()
            # Build the file name
            file_name = '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
                                    conf['latent_dim'], conf["channels"], conf['bs'], conf["activation_func"].__name__,
                                    conf["optimizer"].__name__, conf["loss_func"].__class__.__name__, conf["dropout_rate"], conf["batch_normalization"],
                                    conf['conv_k'],conf['conv_s'],conf['conv_p']
                                    )
            
            # Check if you already trained that combination of parameter
            # Checks in the current output folder or in the results file
            if (not any(file_name in f for f in os.listdir(args.output_dir))) and (not any(file_name in m for m in used_model)):
                print(conf)
                try:
                    #Performs Cross validation
                    validation_accuracy = []
                    for i, (train_index, test_index) in enumerate(kf.split(data['inputs'], data['sequences'])):
                            
                            #Create the dataloader
                            train_loader = DataLoader(list(zip(data['inputs'][train_index], data['sequences'][train_index])), batch_size =  conf['bs'], shuffle=True, num_workers=2, persistent_workers=True, pin_memory=True)
                            test_loader = DataLoader(list(zip(data['inputs'][test_index], data['sequences'][test_index])), batch_size =  conf['bs'], shuffle=False, num_workers=2, persistent_workers=True, pin_memory=True)
                            
                            #Create model
                            model = Autoencoder(conf['output_shape'], conf['channels'], conf['latent_dim'], conv_k=conf['conv_k'],conv_s=conf['conv_s'],conv_p=conf['conv_p'], optimizer=conf['optimizer'], dropout_rate=conf['dropout_rate'], activation_function=conf['activation_func'], loss_func=conf['loss_func'], random_matrices_orth=random_matrices)

                            # define Early stopping callback
                            early_stopping = EarlyStopping(monitor="validation_epoch_loss", patience=3, min_delta=0.0001, verbose=True)

                            #Create the Trainer from PyTorch Litghning. 
                            trainer = pl.Trainer(devices=1, max_epochs=conf['epochs'], callbacks=[early_stopping], num_sanity_val_steps=0, check_val_every_n_epoch=1, enable_progress_bar=False)
                            tuner = pl.tuner.Tuner(trainer)

                            # Finds the best learning rate and updates the model's learning rate (lr) in place. No need to manually change it
                            lr_finder = tuner.lr_find(model,train_dataloaders = train_loader, val_dataloaders=test_loader, update_attr=True)
                            
                            #trains the model
                            trainer.fit(model, train_loader, test_loader)
                            validation_accuracy.append(model.validation_acc[-1])
                            #Save the model
                            # joblib.dump(model, '{}/{}_{}'.format(args.output_dir, file_name,i)) # UNCOMMENT TO SAVE THE MODEL

                            # Save only the last epoch validation accuracy.
                            with open('{}/{}.result'.format(args.output_dir, file_name), 'w') as file: 
                                file.write('{} | {}'.format(np.mean(validation_accuracy), validation_accuracy))
                                file.close()

                except :
                    #ignore the failed model
                    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', type=str, help="Path to the Joblib file containing the data formated in \{inputs, sequences\} (The result of script create_orth_dataset.py)", required=True)
    parser.add_argument('-rom','--rom_path', type=str, help='path to the random orthogonal matrix stored in a JOBLIB file', required=True)
    parser.add_argument('-out', '--output_dir', type=str, help='Folder where to save the models', required=True)
    parser.add_argument('-cv', '--cross_validation', type=int, help='Number of folds for cross validation', default=5, required=False)
    parser.add_argument('-res', '--result_file', type=str, help='Path to the file containing the results of previous experiment. It should be the file produce by "evalutate.py"',required=False, default=None)
    parser.add_argument('-bs', '--batch_size', type=int, help='Batch size')
    parser.add_argument('-ld', '--latent_dim', type=int, help='Latent dimension')

    args = parser.parse_args()
    run(args)