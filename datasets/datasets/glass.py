import numpy as np
import pandas as pd
import os

from ..dataset import Dataset
from .. import utils

class Glass(Dataset):

    name = 'glass'
    reference_website = 'https://archive.ics.uci.edu/ml/datasets/glass+identification'

    def __init__(self):
        
        self._check_download()

        super().__init__(
            features = np.load(os.path.join(self.download_folder, 'features.npy')),
            labels = np.load(os.path.join(self.download_folder, 'labels.npy')),
            y_dim = 6
        )

    def _download(self):

        data_path = utils.download_file('https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data', self.download_folder)
        data = pd.read_csv(data_path, header=None)
        
        features = data.iloc[:, 1:-1].values
        
        data_labels = data.iloc[:, -1]
        label_names = list(data_labels.unique())
        labels = data_labels.map(lambda x : label_names.index(x)).values

        os.remove(data_path)
        
        np.save(os.path.join(self.download_folder, 'features.npy'), features)
        np.save(os.path.join(self.download_folder, 'labels.npy'), labels)