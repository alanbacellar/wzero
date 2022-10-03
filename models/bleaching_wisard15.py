import numpy as np
from .wrappers import CcBleachingWiSARD15
from .model import Model

class BleachingWiSARD15(Model):
    def __init__(self, num_inputs, tuple_lenght, num_classes, canonical=True):
        
        self.__num_inputs = num_inputs
        self.__tuple_lenght = tuple_lenght
        self.__num_classes = num_classes
        self.__canonical = canonical
        
        self.__model = CcBleachingWiSARD15(num_inputs, tuple_lenght, num_classes, canonical)

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
    
    def mental_images(self):
        return self.__model.mental_images()
    
    def get_size(self):
        return self.__model.get_size()