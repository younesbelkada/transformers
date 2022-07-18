import os

class AutoRegressiveDataLoader(object):
    def __init__(self, params) -> None:
        self.path_bin_files = params.path_bin_data
        self.list_bin_files = os.listdir(self.path_bin_files)

    # Load the bin data + iterate to the next one => load dynamically the bin files