# import numpy as np

# from .. import binarization

from .dataset import Dataset
from .dataset import pre_processing

class ImageDataset(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @pre_processing
    def thermometer(self, num_bits, min_=0, max_=255):
        return super().thermometer(num_bits, min_, max_)