import numpy as np
import pandas as pd
import os

from ..dataset import Dataset
from .. import utils

class MagicGammaTelescope(Dataset):

    name = 'magic_gamma_telescope'
    reference_website = 'https://archive.ics.uci.edu/ml/datasets/magic+gamma+telescope'

    def __init__(self):
        
        self._check_download()

        super().__init__(
            features = np.load(os.path.join(self.download_folder, 'features.npy')),
            labels = np.load(os.path.join(self.download_folder, 'labels.npy')),
            label_names = np.load(os.path.join(self.download_folder, 'label_names.npy')),
            y_dim = 2
        )

    def _download(self):

        data_path = utils.download_file('https://archive.ics.uci.edu/ml/machine-learning-databases/magic/magic04.data', self.download_folder)
        data = pd.read_csv(data_path, header=None)
        
        features = data.iloc[:, :-1].values
        
        data_labels = data.iloc[:, -1]
        label_unique = list(data_labels.unique())
        labels = data_labels.map(lambda x : label_unique.index(x)).values
        label_names = np.array(label_unique)

        os.remove(data_path)
        
        np.save(os.path.join(self.download_folder, 'features.npy'), features)
        np.save(os.path.join(self.download_folder, 'labels.npy'), labels)
        np.save(os.path.join(self.download_folder, 'label_names.npy'), label_names)