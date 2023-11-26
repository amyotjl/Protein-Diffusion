import joblib
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import torch
from utils import aa_matrix_encoded

def read_fasta( fasta_path, split_char="!", id_field=0):
    '''
        Reads in fasta file containing multiple sequences.
        Split_char and id_field allow to control identifier extraction from header.
        E.g.: set split_char="|" and id_field=1 for SwissProt/UniProt Headers.
        Returns dictionary holding multiple sequences or only single 
        sequence, depending on input file.
    '''
    
    seqs = dict()
    with open( fasta_path, 'r' ) as fasta_f:
        for line in fasta_f:
            # get uniprot ID from header and create new entry
            if line.startswith('>'):
                uniprot_id = line.replace('>', '').strip().split(split_char)[id_field]
                # replace tokens that are mis-interpreted when loading h5
                uniprot_id = uniprot_id.replace("/","_").replace(".","_")
                seqs[ uniprot_id ] = ''
            else:
                # repl. all whie-space chars and join seqs spanning multiple lines, drop gaps and cast to upper-case
                seq= ''.join( line.split() ).upper().replace("-","")
                # repl. all non-standard AAs and map them to unknown/X
                # seq = seq.replace('U','X').replace('Z','X').replace('O','X')
                seqs[ uniprot_id ] += seq 

    return seqs

def get_input_seqs_dataloader(path, batch_size = 64, shuffle = True, train_size = 0.8, num_workers = 0):
    """
    Creates Dataloaders.

    Parameters : 
        path : path to the random orthogonal matrices (ROM) dataset file
        batch_size 
        shuffle 
        train_size 
        num_workers 

    Returns 
        Train Dataloader and Test Dataloader
    """
    data = joblib.load(path)
    inputs = data['inputs']
    sequences = data['sequences']

    if train_size != 1.0:
        train_inputs, test_inputs, train_seqs, test_seqs = train_test_split(
                                                inputs,
                                                sequences,
                                                train_size=train_size,
                                            )
        train_dataset = list(zip(train_inputs, train_seqs))
        train = DataLoader(train_dataset, batch_size =  batch_size, shuffle=shuffle, num_workers=num_workers)
        
        test_dataset = list(zip(test_inputs, test_seqs))
        test = DataLoader(test_dataset, batch_size =  batch_size,shuffle=shuffle, num_workers=num_workers)
    else:

        train_dataset = list(zip(inputs,sequences))
        train = DataLoader(train_dataset, batch_size =  batch_size, shuffle=shuffle, num_workers=num_workers)
        test = None
    return train, test

# Creates an dataset using a random orthogonal matrix 
def create_orth_dataset(fasta_path, random_orth_matrices_path, min_seq, max_seq, output_file, remove_dups=True):
    """
    Creates an dataset using a random orthogonal matrix. The dataset is stored into a file as a dict{inputs: the matrix, sequences: amino acid sequence} 

    Parameters : 
        fasta_path : path to the fasta file
        random_orth_matrices_path : path to the random orthogonal matrix stored in a JOBLIB file
        min_seq : minimum number of AA in a sequence to be part of the dataset
        max_seq : maximum number of AA in a sequence to be part of the dataset. If it is 0, it is set to the longest sequence
        output_file : path where to output the JOBLIB file
        remove_dups: Whether to remove duplicates or not.
    """
    fasta = read_fasta(fasta_path)
    orth_matrix = joblib.load(random_orth_matrices_path)
    unique = list(set(fasta.values())) if remove_dups else fasta.values()
    seqs = []
    inputs = []
    if max_seq == 0:
        max_seq = max([len(seq) for seq in unique])
    for seq in unique:
        if len(seq) <= max_seq  and len(seq) >= min_seq:
            inputs.append(aa_matrix_encoded(seq, max_seq, orth_matrix, '_'))
            seqs.append(seq)

    joblib.dump({"inputs":torch.stack(inputs), "sequences":seqs}, output_file)
