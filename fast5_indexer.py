import h5py
from glob import glob
import os
import numpy as np
from multiprocessing import Pool, cpu_count
import pickle
from tqdm import tqdm
from pathlib import Path



def get_signal_start_index(signal_array):
    start_point_idx, start_point_val = signal_array[10:3000].argmax(), signal_array[10:3000].max()
    pre_start = signal_array[start_point_idx - 19:start_point_idx - 1]
    pro_start = signal_array[start_point_idx + 1:start_point_idx + 19]
    if len(pro_start) == 0 or len(pre_start) == 0:
        return 0
    if start_point_val > pre_start.mean():
        if start_point_val > pro_start.mean():
            if np.var(signal_array[start_point_idx + 50:start_point_idx + 80]) > np.var(
                    signal_array[start_point_idx - 80:start_point_idx - 50]):
                return start_point_idx
    ## if couldnt find valid start poin then return 0 as start point
    return 0


def is_valid_signal(signal_array):

    sig_start = get_signal_start_index(signal_array)
    return len(signal_array) - sig_start > 5000


def read_multi_fast5(filename):
    with h5py.File(filename, 'r') as f:
        if 'Raw' in f:  # Single read file
            for name, read in f.get('Raw/Reads').items():
                yield name, read.get('Signal')
        else:  # Multi read file
            for name, read in f.items():
                yield name, read.get('Raw/Signal')


def handle_fast5_file(path):
    signals = []
    try:
        for signal_name, signal_array in read_multi_fast5(path):
            if is_valid_signal(signal_array):
                signals.append(signal_name)
    except:
        print('fail ', path)
    return signals


def main():
    import sys
    folder = sys.argv[1]
    files = list(Path(folder).rglob('*.fast5'))
    files = [str(file) for file in files]
    # files = glob('path' + '*.fast5')
    pool = Pool(4)
    results = list(tqdm(pool.imap(handle_fast5_file, files), total=len(files)))
    # results = []
    # for file in tqdm(files, total=len(files)):
    #     try:
    #         results.append(handle_fast5_file(file))
    #     except:
    #         print('fail ', file)
    #         results.append([])
    # print(results)
    res_list = []
    for i, f in enumerate(files):
        for sig in results[i]:
            res_list.append((f, sig))
    with open(os.path.join(folder, 'signal_mapping_tuple.pkl'), 'wb') as f:
        pickle.dump(res_list, f)
    # pool.join()
    # pool.close()

main()
