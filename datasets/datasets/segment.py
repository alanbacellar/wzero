import numpy as np
import pandas as pd
import os

from ..dataset import Dataset
from .. import utils

class Segment(Dataset):

    name = 'segment'
    reference_website = 'https://archive.ics.uci.edu/ml/datasets/image+segmentation'

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

        train_path = utils.download_file('https://archive.ics.uci.edu/ml/machine-learning-databases/image/segmentation.data', self.download_folder)
        data = pd.read_csv(train_path, comment=';')
        
        x_train = data.values
        
        data_labels = data.index
        label_unique = list(data_labels.unique())
        y_train = data_labels.map(lambda x : label_unique.index(x)).values

        os.remove(train_path)

        test_path = utils.download_file('https://archive.ics.uci.edu/ml/machine-learning-databases/image/segmentation.test', self.download_folder)
        data = pd.read_csv(test_path, comment=';')

        x_test = data.values
        
        data_labels = data.index
        y_test = data_labels.map(lambda x : label_unique.index(x)).values

        os.remove(test_path)
        
        np.save(os.path.join(self.download_folder, 'x_train.npy'), x_train)
        np.save(os.path.join(self.download_folder, 'y_train.npy'), y_train)
        np.save(os.path.join(self.download_folder, 'x_test.npy'), x_test)
        np.save(os.path.join(self.download_folder, 'y_test.npy'), y_test)