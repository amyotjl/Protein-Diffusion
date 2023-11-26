import os
import argparse

# Since I trained many autoencoder on many different gpus, the results were scattered. 
# This script collect all results and merges them together

# Read results from previous runs of this script.
# Path : Path to the file containing results from previous runs
def read_previous_results(path):
    results = []
    models = []
    if os.path.exists(path): 
        with open(path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.replace('\n', '')
            results.append(line)
            models.append(line.split(' | ')[-1])
    return results, models


# Merge previous results and new one.
# Reads previous results, then add new one. Output is sorted based on average validation accuracy
def merge_all_results(args):
    results = []
    models_name = []
    if args.results_file:
        results, models_name = read_previous_results(args.results_file)


    for file in os.listdir(args.folder):

        if file.endswith('.result') and file.replace('.result', '') not in models_name:
            with open(os.path.join(args.folder, file), 'r') as f:
                results.append("{} | {}".format(f.readline(), file.replace('.result', '')))
                f.close()
    
    results.sort(key=lambda x : float(x.split()[0]))

    # Write all results to a single file
    with open(args.results_file, 'w') as out:
        for model in results:
            out.write(model)
            out.write('\n')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f','--folder', type=str, help='Folder location to perform the evaluation of the models', required=True)
    parser.add_argument('-res','--results_file', type=str, help='Path to file containing previous results and to update', required=True)

    args = parser.parse_args()
    merge_all_results(args)