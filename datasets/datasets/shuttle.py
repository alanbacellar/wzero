import numpy as np
import pandas as pd
import os
import unlzw3
from pathlib import Path
import io

from ..dataset import Dataset
from .. import utils

class Shuttle(Dataset):

    name = 'shuttle'
    reference_website = 'https://archive.ics.uci.edu/ml/datasets/Statlog+(Shuttle)'

    def __init__(self):
        
        self._check_download()

        super().__init__(
            x_train = np.load(os.path.join(self.download_folder, 'x_train.npy')),
            y_train = np.load(os.path.join(self.download_folder, 'y_train.npy')),
            x_test = np.load(os.path.join(self.download_folder, 'x_test.npy')),
            y_test = np.load(os.path.join(self.download_folder, 'y_test.npy')),
            y_dim = 7
        )

    def _download(self):

        train_path = utils.download_file('https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/shuttle/shuttle.trn.Z', self.download_folder)
        bytes_train_data = unlzw3.unlzw(Path(train_path))
        data = pd.read_csv(io.BytesIO(bytes_train_data), header=None, delim_whitespace=True)
        
        x_train = data.iloc[:, :-1].values
        
        data_labels = data.iloc[:, -1]
        label_unique = list(data_labels.unique())
        y_train = data_labels.map(lambda x : label_unique.index(x)).values

        os.remove(train_path)

        test_path = utils.download_file('https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/shuttle/shuttle.tst', self.download_folder)
        data = pd.read_csv(test_path, header=None, delim_whitespace=True)

        x_test = data.iloc[:, :-1].values
        
        data_labels = data.iloc[:, -1]
        y_test = data_labels.map(lambda x : label_unique.index(x)).values

        os.remove(test_path)
        
        np.save(os.path.join(self.download_folder, 'x_train.npy'), x_train)
        np.save(os.path.join(self.download_folder, 'y_train.npy'), y_train)
        np.save(os.path.join(self.download_folder, 'x_test.npy'), x_test)
        np.save(os.path.join(self.download_folder, 'y_test.npy'), y_test)