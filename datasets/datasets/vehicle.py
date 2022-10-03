import numpy as np
import pandas as pd
import os
import string

from ..dataset import Dataset
from .. import utils

class Vehicle(Dataset):

    name = 'vehicle'
    reference_website = 'https://archive.ics.uci.edu/ml/datasets/Statlog+%28Vehicle+Silhouettes%29'

    def __init__(self):
        
        self._check_download()

        super().__init__(
            features = np.load(os.path.join(self.download_folder, 'features.npy')),
            labels = np.load(os.path.join(self.download_folder, 'labels.npy')),
            label_names = np.load(os.path.join(self.download_folder, 'label_names.npy')),
            y_dim = 6
        )

    def _download(self):

        urls = [f'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/vehicle/xa{i}.dat' for i in string.ascii_lowercase[:9]]

        features = []
        labels = []
        label_names = ['van', 'saab', 'bus', 'opel']

        for url in urls:

            data_path = utils.download_file(url, self.download_folder)
            data = pd.read_csv(data_path, header=None, delim_whitespace=True)
    
            features.append(data.iloc[:, :-1].values)    
            labels_ = data.iloc[:, -1]
            labels.append(labels_.map(lambda x : label_names.index(x)).values)

            os.remove(data_path)
        
        features = np.concatenate(features, axis=0)
        labels = np.concatenate(labels, axis=0)
        label_names = np.array(label_names)

        np.save(os.path.join(self.download_folder, 'features.npy'), features)
        np.save(os.path.join(self.download_folder, 'labels.npy'), labels)
        np.save(os.path.join(self.download_folder, 'label_names.npy'), label_names)