import numpy as np
from .wrappers import CcProbWiSARD
from .model import Model

class ProbWiSARD(Model):
    def __init__(self, num_inputs, tuple_lenght, num_classes, canonical=True):
        
        self.__num_inputs = num_inputs
        self.__tuple_lenght = tuple_lenght
        self.__num_classes = num_classes
        self.__canonical = canonical
        
        self.__model = CcProbWiSARD(num_inputs, tuple_lenght, num_classes, canonical)

        self.hyperparameters = {
            'tuple_lenght': tuple_lenght,
            'canonical': canonical
        }

    @property
    def num_inputs(self):
        return self.__num_inputs
    
    @property
    def tuple_lenght(self):
        return self.__tuple_lenght
    
    @property
    def num_classes(self):
        return self.__num_classes
    
    @property
    def canonical(self):
        return self.__canonical

    def train(self, x, y):
        self.__model.train(x, y)

    def predict(self, x):
        return self.__model.predict(x)