import numpy as np
import json
import os
import argparse

def merge_diffusion_results(args):
    results = {}
    diffusion_folder = args.folder
    result_file = 'result.json'

    #Gather all subfolder that represent all trained models.
    folders = [folder for folder in os.listdir(diffusion_folder) if os.path.isdir(os.path.join(diffusion_folder, folder))]

    print(len(folders))
    for folder in folders:

        #If there is a precomputed result file, we fetch the content. Otherwise, we go to the next folder
        if result_file in os.listdir('{}/{}'.format(diffusion_folder, folder)):

            #Open the file and get the FIDs
            with open(os.path.join(diffusion_folder, folder, result_file)) as f:
                res = json.load(f)

            fids = res['fids'] if 'fids' in res else [-1]

            # Fetch the config file. Not needed, but helps a bit to manually see the config of the good results
            with open('{}/{}/config.json'.format(diffusion_folder, folder)) as f:
                config = json.load(f)

            results[folder] = {'fids': fids, 'config':config}
    print("{} models founds. Merging them.".format(len(results)))
    # Output the results.
    with open('{}/{}'.format(diffusion_folder, args.output_file), 'w') as f:
        for item in sorted(results.items(), key=lambda item : item[1]['fids'][-1]): # sort by the last checkpoint FID
            f.write("{} | {} | {}\n".format(item[1]['fids'], item[0], item[1]['config']))

        f.close()
    print('DONE')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder', type=str, help='Root folder where all model where saved', required=True)
    parser.add_argument('-out', '--output_file', type=str, help='Name of the JSON file to write the output all results ', default='all_result')
    args = parser.parse_args()

    merge_diffusion_results(args)
