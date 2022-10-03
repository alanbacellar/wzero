import numpy as np
import itertools
import copy

from .experiment import DatasetExperiment

class Preprocessings(DatasetExperiment):
    def __init__(self, **preprocessings):
        super().__init__()

        self.preprocessings = preprocessings
        self.preprocessings_names = list(preprocessings.keys())
        self.preprocessings_values = list(self.preprocessings.values())

        # Make sure every value is an iterable in order to itertools.product to work
        for preprocessing_name in self.preprocessings_names:
            if not hasattr(self.preprocessings[preprocessing_name], '__iter__'):
                self.preprocessings[preprocessing_name] = [self.preprocessings[preprocessing_name]]

    def call(self, dataset, model):

        data = []
                
        for preprocessing_args in itertools.product(*self.preprocessings.values()):
            
            iteration_dataset = copy.deepcopy(dataset)
            
            for preprocessing_name, preprocessing_arg in zip(self.preprocessings_names, preprocessing_args):
                
                if preprocessing_arg is None:
                    getattr(iteration_dataset, preprocessing_name)()
                else:
                    getattr(iteration_dataset, preprocessing_name)(preprocessing_arg)
                
            data.extend(self._next(iteration_dataset, model))

        return data