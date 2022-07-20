import os
import pickle
import random

from functools import partial
from multiprocessing import Pool
from regex import R

from torch.utils import data
import jax.numpy as np


def numpy_collate(batch, max_seq_len):
    batch = truncate_and_extend(batch, max_seq_len)
    return np.array(batch)

# TODO: optimize this
def replace_by_another_string(batch, max_seq_len):
    new_batch = []
    for b in batch:
        while len(b) < max_seq_len:
            random_index = random.randint(0, len(batch))
            b.extend(batch[random_index])
        new_batch.append(b)
    return new_batch


def truncate_and_extend(batch, max_seq_len):
    batch = replace_by_another_string(batch, max_seq_len)
    truncated_batch = list(map(lambda x: x[:max_seq_len], batch))
    return truncated_batch
    
class AutoRegressiveDataset(data.Dataset):
    def __init__(self, params):
        self.path_bin_files = params.path_bin_data
        self.list_bin_files = os.listdir(self.path_bin_files)

        self.max_seq_len = params.max_seq_len
        self.truncate_func = partial(numpy_collate, max_seq_len=self.max_seq_len)

        self.current_file_index = 0
        bin_file = open(os.path.join(self.path_bin_files, self.list_bin_files[self.current_file_index]), "rb")
        self.current_data = pickle.load(bin_file)
        # Load the bin data + iterate to the next one => load dynamically the bin files
    
    def _update_current_file(self):
        self.current_file_index += 1
        bin_file = open(os.path.join(self.path_bin_files, self.list_bin_files[self.current_file_index]), "rb")
        self.current_data = pickle.load(bin_file)

    
    def __len__(self):
        return len(self.list_bin_files * 2048)
    
    
    def __getitem__(self, idx):
        try:
            data = self.current_data[idx]
        except:
            self._update_current_file()
            data = self.current_data[idx]
        return data

class AutoRegressiveDataLoader(data.DataLoader):
    def __init__(self, dataset, batch_size, shuffle=False, sampler=None, num_workers=16):
        super(self.__class__, self).__init__(dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=dataset.truncate_func,)