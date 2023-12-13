import torch
from torch.utils.data import Dataset
from torch import Tensor
from denoising_diffusion_pytorch import Trainer1D, Unet1D, GaussianDiffusion1D
import joblib
import glob
import os
import re
import json
import argparse
import hashlib
from utils import get_combinations, hash_dictionary


def hash_dictionary(dictionary):
    # Convert the dictionary to a JSON string
    json_str = json.dumps(dictionary, sort_keys=True)

    # Create a hash object using the SHA256 algorithm
    hash_object = hashlib.sha256(json_str.encode())

    # Get the hexadecimal representation of the hash
    hash_hex = hash_object.hexdigest()

    return hash_hex


class Dataset1D(Dataset):
    def __init__(self, tensor: Tensor):
        super().__init__()
        self.tensor = tensor.clone()

    def __len__(self):
        return len(self.tensor)

    def __getitem__(self, idx):
        return self.tensor[idx].clone()


#Returns the dictionary without specified keys
def without_keys(d, keys):
    return {k: d[k] for k in d.keys() - keys}

def train(config, data, args):
    # creating Dataset1D
    dataset_train = Dataset1D(data['inputs'].detach())

    #Hashing the config and creating the folder
    conf_to_hash_unet = without_keys(config['unet'], [])
    conf_to_hash_gaussian = without_keys(config['gaussian'], [])
    conf_to_hash_trainer = without_keys(config['trainer'], ['train_batch_size', 'train_num_steps', 'save_and_sample_every', 'num_samples'])

    folder_name ='{}/{}'.format(args.output_dir,hash_dictionary( {**conf_to_hash_unet, **conf_to_hash_gaussian,**conf_to_hash_trainer}))


    #Create the result folder for the current configuration
    try:
        os.makedirs(folder_name, exist_ok=False)
    except Exception:

        print('ALREADY TRAINED ')
        return


    # Dump the configuration into a JSON file
    with open('{}/config.json'.format(folder_name), 'w') as file:
        json.dump(config, file, indent=4)

    print(folder_name)


    #creating diffusion model
    model = Unet1D(
        **config['unet']
    ).cuda()

    diffusion = GaussianDiffusion1D(
        model,
        **config['gaussian']
    ).cuda()

    trainer = Trainer1D(
        diffusion,
        dataset = dataset_train,
        results_folder=folder_name,
        **config['trainer']
    )


    # If the training was interupted, we load the last checkpoint
    list_of_files = glob.glob('{}/*.pt'.format(folder_name)) # * means all if need specific format then *.csv
    if len(list_of_files)>0:
        latest_file = max(list_of_files, key=os.path.getctime)
        print(latest_file)
        pattern = r"-(.*?)\."
        match = re.search(pattern, latest_file)
        if match:
            result = match.group(1)
            print('Loading checkpoint: ', result)
            trainer.load(result)

    trainer.train()

#returns all combination of a given dictionary where values are lists
def get_diffusion_combinations(config):
    temp = {}
    for key in config.keys():
        temp[key] = get_combinations(config[key])
    return get_combinations(temp)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-data', type=str, help="Path to the Joblib file containing the LATENT data formated in \{inputs, sequences\} (The result of script create_orth_dataset.py)", required=True)
    parser.add_argument('-out', '--output_dir', type=str, help='Folder where to save the models', required=True)
    parser.add_argument('-d', '--dim', type=int, help='inital dimension of the model', default=16)

    args = parser.parse_args()


    conf = {
    'trainer':{
        'train_batch_size' : 128,
        'gradient_accumulate_every' : 2,
        'train_lr' : [0.00001],
        'train_num_steps' : 10000,
        'ema_update_every' : 10,
        'ema_decay' : 0.995,
        'adam_betas' : (0.9, 0.99),
        'save_and_sample_every' : 2000,
        'num_samples' : 1, # must be square
        'amp' : False,
        'split_batches' : True,
    },
    'unet':{
        'dim':int(args.dim),
        'dim_mults':[(1, 2, 4, 8, 16)],
        'channels':1,
        'out_dim' : None,
        'self_condition' : [True],
        'resnet_block_groups' : [4],
        'learned_variance' : False,
        'learned_sinusoidal_cond' : False,
        'random_fourier_features' : False,
        'learned_sinusoidal_dim' : [16]
    },
    'gaussian':{
        'seq_length': 1024,
        'timesteps' : 1000,
        'sampling_timesteps' : None,
        'objective' : ['pred_x0'],
        'beta_schedule' : ['cosine', 'linear'],
        'ddim_sampling_eta' : 0.,
        'auto_normalize' : True
    },

}

    dataset = joblib.load(args.data)
    for config in get_diffusion_combinations(conf):
        print(config)
        train(config, dataset, args)
