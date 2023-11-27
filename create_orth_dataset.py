from load_dataset import create_orth_dataset
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-fp','--fasta_path', type=str, help="Path to the fasta data file", required=True)
    parser.add_argument('-romp','--rom_path', type=str, help='path to the random orthogonal matrix stored in a JOBLIB file', required=True)
    parser.add_argument('-out','--output_file', type=str, help='path where to output the JOBLIB file', required=True)
    parser.add_argument('-mis', '--min_seq', type=int, help='minimum number of AA in a sequence to be part of the dataset', default=0, required=False)
    parser.add_argument('-mas', '--max_seq', type=int, help='maximum number of AA in a sequence to be part of the dataset', default=0) # gets set in the function
    parser.add_argument('-rd', '--remove_dups', type=bool, help='Whether to remove duplicates or not.', default=True, required=False)


    args = parser.parse_args()

    create_orth_dataset(fasta_path=args.fasta_path,
                        random_orth_matrices_path=args.rom_path,
                        min_seq=args.min_seq,
                        max_seq=args.max_seq,
                        output_file= args.output_file,
                        remove_dups= args.remove_dups)
