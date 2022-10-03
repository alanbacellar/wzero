import numpy as np
from pathlib import Path
import os
import functools
import inspect
from ..utils import utils

from .. import binarization

def pre_processing(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        
        # We only want to add to pre_processing calls made by the user, not internal calls.
        set_internal = False
        if not self._internal:
            self._internal = True
            set_internal = True
        
        try:
            output = func(self, *args, **kwargs)
        except Exception as e:
            if set_internal:
                self._internal = False
                raise e

        # We add to pre_processings if func call is sucessfull
        if set_internal:

            # Get argnames and values passed to function and it's defaults
            ###########################################
            # arg_names = func.__code__.co_varnames[1:] # skip self
            arg_names = inspect.getargspec(func).args[1:]
            all_args_dict= {arg_name:arg for arg_name, arg in zip(arg_names, args)}
            all_args_dict.update(kwargs)

            if func.__defaults__ is not None:
                num_kwargs = len(func.__defaults__)
                num_args = len(arg_names) - num_kwargs

                for arg_name in arg_names:
                    if arg_name not in all_args_dict:
                        all_args_dict[arg_name] = func.__defaults__[arg_names.index(arg_name)-num_args]
            ###########################################
            
            self.pre_processings.append((func.__name__, all_args_dict))
            self._internal = False
        
        return output
    
    return wrapper

class Dataset:
    def __init__(self, x_train=None, y_train=None, x_test=None, y_test=None, features=None, labels=None, label_names=None, y_dim=None, name=None):
        
        assert ((type(x_train) != type(None) and type(y_train) != type(None) and type(x_test) != type(None) and type(y_test) != type(None)) or 
                (type(features) != type(None) and type(labels) != type(None))), '(x_train, y_train ,x_test and y_test) or (features and labels) needs to be set'
        
        if type(x_train) != type(None):
            self.x_train = x_train
            self.y_train = y_train
            self.x_test = x_test
            self.y_test = y_test
            self.splitted = True
            self.x_dim = np.array(self.x_train.shape[1:])
        else:
            self.features = features
            self.labels = labels
            self.splitted = False
            self.x_dim = np.array(self.features.shape[1:])

        self.label_names = label_names
        self.y_dim = y_dim

        self.name = self.__class__.__name__.lower()
        if self.name == 'dataset':
            self.name = 'user_dataset'
        if name is not None:
            self.name = name
        
        self.pre_processings = []
        self._internal = False

    @property
    def download_folder(self):
        return self.__download_folder
    
    @property
    def downloaded_datasets(self):
        return self.__downloaded_datasets

    def _check_download(self):
        self.__main_download_folder = os.path.join(Path.home(), 'wzero_datasets')
        os.makedirs(self.__main_download_folder, exist_ok=True)
        self.__downloaded_datasets = os.listdir(self.__main_download_folder)
        self.__download_folder = os.path.join(self.__main_download_folder, self.name) # downlaod folder of the child dataset; self.name -> static attribute of child dataset
        if not self.name in self.__downloaded_datasets:
            os.makedirs(self.__download_folder, exist_ok=True)
            self._download()

    @pre_processing
    def flatten(self):
        
        if self.splitted:
            self.x_train = self.x_train.reshape(-1, self.x_dim.prod())
            self.x_test = self.x_test.reshape(-1, self.x_dim.prod())
            self.x_dim = np.array(self.x_train.shape[1:])
        else:
            self.features = self.features.reshape(-1, self.x_dim.prod())
            self.x_dim = np.array(self.features.shape[1:])

        return self

    @pre_processing
    def join(self):

        if not self.splitted:
            return self

        self.features = np.concatenate([self.x_train, self.x_test], axis=0)
        self.labels = np.concatenate([self.y_train, self.y_test], axis=0)
        self.splitted = False

        return self
    
    @pre_processing
    def train_test_split(self, train_size=None, split_index=None, shuffle=True, seed=None):
        
        if self.splitted:
            return self
        
        assert train_size != None or split_index != None, "train_size or split_index needs to be set"

        if shuffle:
            self.shuffle(seed)

        if train_size != None:
            split_index = int(self.features.shape[0] * train_size)

        self.x_train = self.features[:split_index]
        self.y_train = self.labels[:split_index]
        self.x_test = self.features[split_index:]
        self.y_test = self.labels[split_index:]

        del self.features
        del self.labels

        self.splitted = True

        return self
    
    @pre_processing
    def shuffle(self, seed=None, concat_shuffle=False):
        
        if concat_shuffle and self.splitted:
            train_size = self.x_train.shape[0]
            self.join()
            self.train_test_split(split_index=train_size, shuffle=True)
        elif self.splitted:
            self.x_train, self.y_train = utils.joint_shuffle(self.x_train, self.y_train, seed=seed)
            self.x_test, self.y_test = utils.joint_shuffle(self.x_test, self.y_test, seed=seed)
        else:
            self.features, self.labels = utils.joint_shuffle(self.features, self.labels, seed=seed)

        return self
    
    @pre_processing
    def threshold(self, t):
        if self.splitted:
            self.x_train = (self.x_train > t)
            self.x_test = (self.x_test > t)
        else:
            self.features = (self.features > t)
        return self

    @pre_processing
    def thermometer(self, num_bits, min_=None, max_=None):
        if self.splitted: 
            min_ = self.x_train.min(axis=0) if min_ == None else min_
            max_ = self.x_train.max(axis=0) if max_ == None else max_
            self.x_train = binarization.thermometer(self.x_train, num_bits, min_, max_)
            self.x_test = binarization.thermometer(self.x_test, num_bits, min_, max_)
            self.x_dim = np.array(self.x_train.shape[1:])
        else:
            min_ = self.features.min(axis=0) if min_ == None else min_
            max_ = self.features.max(axis=0) if max_ == None else max_
            self.features = binarization.thermometer(self.features, num_bits, min_, max_)
            self.x_dim = np.array(self.features.shape[1:])
            
        return self
    
    @pre_processing
    def gaussian_thermometer(self, num_bits, individual=True):
        if self.splitted: 
            thresholds = binarization.gaussian_thresholds(self.x_train, num_bits, individual=individual)
            # print(thresholds)
            self.x_train = binarization.gaussian_thermometer(self.x_train, thresholds=thresholds)
            self.x_test = binarization.gaussian_thermometer(self.x_test, thresholds=thresholds)
            self.x_dim = np.array(self.x_train.shape[1:])
        else:
            self.features = binarization.gaussian_thermometer(self.features, num_bits, individual=individual)
            self.x_dim = np.array(self.features.shape[1:])
        return self

    @pre_processing
    def exponential_thermometer(self, num_bits, individual=True):
        if self.splitted: 
            thresholds = binarization.exponential_thresholds(self.x_train, num_bits, individual=individual)
            # print(thresholds)
            self.x_train = binarization.exponential_thermometer(self.x_train, thresholds=thresholds)
            self.x_test = binarization.exponential_thermometer(self.x_test, thresholds=thresholds)
            self.x_dim = np.array(self.x_train.shape[1:])
        else:
            self.features = binarization.exponential_thermometer(self.features, num_bits, individual=individual)
            self.x_dim = np.array(self.features.shape[1:])
        return self
    
    @pre_processing
    def expnorm_thermometer(self, num_bits, individual=True):
        if self.splitted: 
            thresholds = binarization.expnorm_thresholds(self.x_train, num_bits, individual=individual)
            # print(thresholds)
            self.x_train = binarization.expnorm_thermometer(self.x_train, thresholds=thresholds)
            self.x_test = binarization.expnorm_thermometer(self.x_test, thresholds=thresholds)
            self.x_dim = np.array(self.x_train.shape[1:])
        else:
            self.features = binarization.expnorm_thermometer(self.features, num_bits, individual=individual)
            self.x_dim = np.array(self.features.shape[1:])
        return self
    
    @pre_processing
    def distrib_thermometer(self, num_bits, individual=True):
        if self.splitted:
            splits = binarization.get_splits(self.x_train, num_bits, individual=individual)
            self.x_train = binarization.distrib_therm(self.x_train, splits=splits)
            self.x_test = binarization.distrib_therm(self.x_test, splits=splits)
            self.x_dim = np.array(self.x_train.shape[1:])
        else:
            self.features = binarization.distrib_therm(self.features, num_bits, individual=individual)
            self.x_dim = np.array(self.features.shape[1:])
        
        return self

    @pre_processing
    def distrib_thermometer2(self, num_bits, individual=True):
        if self.splitted:
            splits = binarization.get_splits2(self.x_train, self.y_train, num_bits)
            self.x_train = binarization.distrib_therm2(self.x_train, splits=splits)
            self.x_test = binarization.distrib_therm2(self.x_test, splits=splits)
            self.x_dim = np.array(self.x_train.shape[1:])
        else:
            raise "Dataset not splitted for distrib therm 2"
        return self
    
    def label_distribution(self):

        if self.splitted:

            train_dist = np.array([(self.y_train == i).sum() for i in range(self.y_dim)])
            train_dist = train_dist / train_dist.sum()

            test_dist = np.array([(self.y_test == i).sum() for i in range(self.y_dim)])
            test_dist = test_dist / test_dist.sum()

            data = {
                'train': train_dist,
                'test': test_dist
            }
            
            return data
        
        else:

            dist = np.array([(self.labels == i).sum() for i in range(self.y_dim)])
            dist = dist / dist.sum()

            data = {
                'labels': dist
            }
            
            return data


