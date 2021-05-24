from random import shuffle as shuffle_func
from torch.utils.data.dataset import Dataset
import h5py
import numpy as np
import torch

CACHE_SIZE = 2048


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


# gets a dict
class NanoporeDataset(Dataset):

    def __init__(self, data_dict, transform=None):
        """
        note: Always shuffles, dict has now order
        @param data_dict: {label: signal_paths} each signal path is h5path:read_id
        @param shuffle:
        @param transform:
        """
        self.label_options = list(data_dict.keys())
        self._all_signals = []
        for label in self.label_options:
            self._all_signals += [(path, read_id, label) for path, read_id in data_dict[label]]
        shuffle_func(self._all_signals)
        self.transform = transform
        self.numOfReadsLeft = len(self._all_signals)

    def _get_signal(self, path, read_id):
        with h5py.File(path, 'r') as f:
            signal = f.get(read_id).get('Raw/Signal')
            start_point = get_signal_start_index(signal)
            return signal[start_point:start_point + 5000]

    def __getitem__(self, index):
        path, read_id, label = self._all_signals[index]
        signal = self._get_signal(path, read_id)
        if self.transform is not None:
            signal = self.transform(signal)
        return torch.from_numpy(signal).type(torch.FloatTensor), label

    def __len__(self):
        return len(self._all_signals)
