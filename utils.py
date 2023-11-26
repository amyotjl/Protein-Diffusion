import torch
import numpy as np
import itertools
from scipy import spatial as sp
import hashlib
import json

from scipy import linalg


def hardmax(matrix, device):
    max_indices = torch.argmax(matrix, axis=1)  # Find the indices of the maximum values along each row
    hardmax_matrix = torch.zeros_like(matrix)  # Create a matrix of zeros with the same shape as the input matrix
    row_indices = torch.arange(matrix.shape[0])  # Generate row indices

    hardmax_matrix[row_indices, max_indices] = 1  # Set the values at the maximum indices to 1

    return hardmax_matrix.to(device)

# Function to one-hot encode a sequence of amino acid.
# The output is a matrix of max_length_amino_acid (1965) x unique_amino_acid (21). Sequence shorter than max_length_amino_acid are filled with 0. 
def one_hot_encode(seq, max_length, unique, padding_char = None):
    if padding_char != None:
        seq = seq.ljust(max_length, padding_char)
        unique.append(padding_char)
    matrix = np.zeros((max_length, len(unique)))
    for idx, elem in enumerate(seq):
        matrix[idx][unique.index(elem)] = 1
    return matrix.astype(np.float32)

def aa_matrix_encoded(seq, max_length, aa_vectors, padding_char = None):
    if padding_char != None:
        seq = seq.ljust(max_length, padding_char)
        
    matrix = torch.empty((max_length, len(aa_vectors)))
    for idx, elem in enumerate(seq):
        matrix[idx] = aa_vectors[elem]
    return matrix


def generate_random_matrices(unique_aa):
    res = {}
    for aa in unique_aa:
        res[aa] = torch.randn(len(unique_aa))
    return res


def mul_random_matrix(seq, max_length, random_matrices):
    seq = seq.ljust(max_length, '_')
    matrix = torch.empty((max_length, len(random_matrices.keys())))
    for idx, elem in enumerate(seq):
        matrix[idx] = random_matrices[elem]
    return matrix

def reconstruct_sequence(AA_random_matrices, output):
    tree = sp.KDTree(torch.stack([vec for vec in AA_random_matrices.values()]))
    closest = tree.query(output)[1]
    aminoacid = list(AA_random_matrices.keys())
    reconstructed = []
    # vectors = np.array(list(AA_random_matrices.values()))
    for idx, row in enumerate(output):
        reconstructed.append([aminoacid[i] for i in closest[idx]])

    return reconstructed


def get_combinations(dictionary):
    for key, value in dictionary.items():
        if not isinstance(value, list):
            dictionary[key] = [value]
    value_lists = dictionary.values()
    combinations = list(itertools.product(*value_lists))
    keys = dictionary.keys()
    
    result = []
    for combination in combinations:
        combined_dict = dict(zip(keys, combination))
        result.append(combined_dict)
    return result


    # Call the recursive function to get all combinations
    all_combinations = recursive_combinations(temp)

    return all_combinations

def correct_reconstructed_amino_acid(sequence, output, AA_random_matrices):
    reconstructed = list(reconstruct_sequence(AA_random_matrices, output))
    ground_truth = list(sequence.ljust(1965,'_'))
    return sum(x == y for x, y in zip(reconstructed, ground_truth))

def batch_correct_reconstructed_amino_acid(sequences, output, AA_random_matrices, longest_sequence):
    tree = sp.KDTree(torch.stack([vec for vec in AA_random_matrices.values()]))
    closest = tree.query(output)[1]
    aminoacid = list(AA_random_matrices.keys())
    correct_aa = 0
    reconstructed_pair = []
    for idx, seq in enumerate(sequences):
        reconstructed = [aminoacid[i] for i in closest[idx]]
        seq = list(seq.ljust(longest_sequence, '_'))
        reconstructed_pair.append((seq, reconstructed))
        correct_aa += sum(x == y for x, y in zip(reconstructed, seq))
    accuracy = correct_aa/ (len(sequences)*longest_sequence)
    return correct_aa, accuracy, reconstructed_pair
    
def hash_dictionary(dictionary):
    # Convert the dictionary to a JSON string
    json_str = json.dumps(dictionary, sort_keys=True)
    
    # Create a hash object using the SHA256 algorithm
    hash_object = hashlib.sha256(json_str.encode())
    
    # Get the hexadecimal representation of the hash
    hash_hex = hash_object.hexdigest()
    
    return hash_hex

def compute_stats(data):
    mu = np.mean(data, axis = 0)
    sigma = np.cov(data, rowvar=False)
    return mu, sigma

def fid(mur, muf, sigr, sigf, eps = 1e-6):

    mur = np.atleast_1d(mur)
    muf = np.atleast_1d(muf)

    sigr = np.atleast_2d(sigr)
    sigf = np.atleast_2d(sigf)

    diff = mur-muf
    covmean, _ = linalg.sqrtm(sigr.dot(sigf), disp=False)

    if not np.isfinite(covmean).all():
        print('product of cov matrices is singular')
        offset = np.eye(sigf.shape[0]) * eps
        covmean = linalg.sqrtm((sigf + offset) @ (sigr + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f'Imaginary component {m}')
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    return diff.dot(diff) + np.trace(sigr) + np.trace(sigf) - 2 * tr_covmean

def without_keys(d, keys):
    return {k: d[k] for k in d.keys() - keys}