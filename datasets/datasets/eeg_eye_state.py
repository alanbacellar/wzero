import numpy as np
import pandas as pd
import os

from ..dataset import Dataset
from .. import utils

class EEGEyeState(Dataset):

    name = 'eeg_eye_state'
    reference_website = 'https://archive.ics.uci.edu/ml/datasets/EEG+Eye+State'

    def __init__(self):
        
        self._check_download()

        super().__init__(
            features = np.load(os.path.join(self.download_folder, 'features.npy')),
            labels = np.load(os.path.join(self.download_folder, 'labels.npy')),
            y_dim = 2
        )

    def _download(self):

        data_path = utils.download_file('https://archive.ics.uci.edu/ml/machine-learning-databases/00264/EEG%20Eye%20State.arff', self.download_folder)
        data = pd.read_csv(data_path, comment='@', header=None)
        
        features = data.iloc[:, :-1].values
        
        data_labels = data.iloc[:, -1]
        label_names = list(data_labels.unique())
        labels = data_labels.map(lambda x : label_names.index(x)).values

        os.remove(data_path)
        
        np.save(os.path.join(self.download_folder, 'features.npy'), features)
        np.save(os.path.join(self.download_folder, 'labels.npy'), labels)