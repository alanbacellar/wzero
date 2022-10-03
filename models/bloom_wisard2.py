import numpy as np
from .wrappers import CcBloomWiSARD2
from .model import Model

class BloomWiSARD2(Model):
    def __init__(self, num_inputs, tuple_lenght, num_filters, filter_tuple_lenght, num_classes):
        super().__init__()

        self.__num_inputs = num_inputs
        self.__tuple_lenght = tuple_lenght
        self.__num_filters = num_filters
        self.__filter_tuple_lenght = filter_tuple_lenght
        self.__num_classes = num_classes
        
        self.__model = CcBloomWiSARD2(num_inputs, tuple_lenght, num_filters, filter_tuple_lenght, num_classes)

        self.hyperparameters = {
            'tuple_lenght': tuple_lenght,
            'num_filters': num_filters,
            'filter_tuple_lenght': filter_tuple_lenght,
        }

    @property
    def num_inputs(self):
        return self.__num_inputs
    
    @property
    def tuple_lenght(self):
        return self.__tuple_lenght
    
    @property
    def num_filters(self):
        return self.__num_filters

    @property
    def filter_tuple_lenght(self):
        return self.__filter_tuple_lenght
    
    @property
    def num_classes(self):
        return self.__num_classes

    def train(self, x, y):
        self.__model.train(x, y)

    def predict(self, x):
        return self.__model.predict(x)