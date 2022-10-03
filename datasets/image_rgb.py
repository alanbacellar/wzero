import numpy as np

from .image import ImageDataset

class ImageDatasetRGB(ImageDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grayscaled = False

    def grayscale(self, weights=np.ones(3)):
        self.x_train = self.x_train.mean(axis=-1)
        self.x_test = self.x_test.mean(axis=-1)
        self.x_dim = self.x_train.shape[1:]
        return self
