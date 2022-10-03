import numpy as np
import copy

from .experiment import DatasetExperiment, reduce_datas

class KFold(DatasetExperiment):
    def __init__(self, k=5, reduce='mean', shuffle=True, seed=None, skip_splitted=False, inverted=False):
        super().__init__()

        if type(k) is not int:
            raise ValueError("")
        if k <= 0:
            raise ValueError("")
        self.k = k

        if reduce not in (None, 'mean', 'median', 'max', 'min'):
            raise ValueError("")
        self.reduce = reduce

        if type(shuffle) is not bool:
            raise ValueError("")
        self.shuffle = shuffle

        if not(type(seed) is int or seed is None):
            raise ValueError("")
        self.seed = seed

        if type(skip_splitted) is not bool:
            raise ValueError("")
        self.skip_splitted = skip_splitted

        if type(inverted) is not bool:
            raise ValueError("")
        self.inverted = inverted

    
    def call(self, dataset, model):

        if dataset.splitted:
            if self.skip_splitted:
                return self._next(dataset, model)
            else:
                dataset.join()
        
        if self.shuffle:
            if self.seed is not None:
                dataset.shuffle(self.seed)
            else:
                dataset.shuffle()

        fold_dataset = copy.copy(dataset)
        
        size = fold_dataset.features.shape[0]
        k_size = size // self.k
        k_mod = size % self.k

        fold_dataset.splitted = True
        del fold_dataset.features
        del fold_dataset.labels

        w = np.array([k_size + int(i < k_mod) for i in range(self.k)])

        fold_datas = []

        begin = 0
        for i in range(self.k):
            
            end = begin + w[i]
            
            fold_dataset.x_train = dataset.features[begin:end]
            fold_dataset.y_train = dataset.labels[begin:end]

            fold_dataset.x_test = np.concatenate([dataset.features[:begin], dataset.features[end:]], axis=0)
            fold_dataset.y_test = np.concatenate([dataset.labels[:begin], dataset.labels[end:]], axis=0)

            if self.inverted:
                fold_dataset.x_train, fold_dataset.x_test = fold_dataset.x_test, fold_dataset.x_train
                fold_dataset.y_train, fold_dataset.y_test = fold_dataset.y_test, fold_dataset.y_train

            begin = end

            fold_data = self._next(fold_dataset, model)
            fold_datas.append(fold_data)


        return reduce_datas(self.reduce, fold_datas, w, experiment_name='kfold')