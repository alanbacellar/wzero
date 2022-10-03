import numpy as np

from .experiment import ModelExperiment, reduce_datas

class Mappings(ModelExperiment):
    def __init__(self, num_mappings=10, reduce='mean'):
        super().__init__()

        self.num_mappings = num_mappings

        if reduce not in (None, 'mean', 'median', 'max', 'min'):
            raise ValueError("")

        self.reduce = reduce

    def call(self, dataset, model):
        
        mappings_datas = [self._next(dataset, model) for i in range(self.num_mappings)]
        
        return reduce_datas(self.reduce, mappings_datas, w=np.ones(self.num_mappings), experiment_name='mappings')