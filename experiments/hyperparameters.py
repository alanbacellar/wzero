import numpy as np
import itertools

from .experiment import ModelExperiment

class Hyperparameters(ModelExperiment):
    def __init__(self, **hyperparameters):
        super().__init__()

        self.hyperparameters = hyperparameters
        self.hyperparemeters_names = list(hyperparameters.keys())
        self.hyperparemeters_values = list(self.hyperparameters.values())

        # Make sure every value is an iterable in order to itertools.product to work
        for hyperparemeter_name in self.hyperparemeters_names:
            if not hasattr(self.hyperparameters[hyperparemeter_name], '__iter__'):
                self.hyperparameters[hyperparemeter_name] = [self.hyperparameters[hyperparemeter_name]]

    def call(self, dataset, model):

        data = []
        
        for hyperparemeter_args in itertools.product(*self.hyperparemeters_values):
            
            iteration_hyperparemeters = dict(zip(self.hyperparemeters_names, hyperparemeter_args))
            iteration_model = lambda **kwargs : model(**iteration_hyperparemeters, **kwargs)

            data.extend(self._next(dataset, iteration_model))

        return data