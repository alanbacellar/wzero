import numpy as np
import functools
import types
import itertools
import copy
from abc import abstractclassmethod

from .. import metrics
from ..datasets.dataset import Dataset
from ..models.model import Model

def data_creator(dataset, model):

    # data = [{
    #     'info': {
    #         'dataset': dataset.name,
    #         'model':  model.__class__.__name__.lower(),
    #         'hyper_parameters': model.hyperparameters
    #     },
    #     'metric': {}
    # }]

    # pre = dataset.pre_processings
    # for p in pre:
    #     key = p[0]
    #     value = p[1][list(p[1].keys())[0]]
    #     data['info'][key] = value


    data = [{
        'info': {
            'dataset': dataset.name,
            'pre_processings': dataset.pre_processings,
            'model':  model.__class__.__name__.lower(),
            'hyper_parameters': model.hyperparameters
        },
        'metric': {}
    }]
    
    return data

metrics_dict = {
    'accuracy': metrics.accuracy
}

def reduce_datas(reduce, datas, w=None, experiment_name=None):

    n = len(datas)

    reduced_data = datas[0]

    if w is not None:
        if type(w) is not np.ndarray:
            raise ValueError("")
        size = w.sum()

    for j in range(len(datas[0])):
        for i in range(n):
            for metric in datas[0][j]['metric']:
                
                if i == 0:
                    tmp = reduced_data[j]['metric'][metric]
                    if type(tmp) is not np.ndarray:
                        reduced_data[j]['metric'][metric] = np.zeros(n)
                    else:
                        reduced_data[j]['metric'][metric] = np.zeros([n, *tmp.shape])
                    reduced_data[j]['metric'][metric][0] = tmp
                else:
                    reduced_data[j]['metric'][metric][i] = datas[i][j]['metric'][metric]

    for j in range(len(datas[0])):
        for metric in list(datas[0][j]['metric'].keys()):
            
            if 'std' in metric:
                continue
            
            if reduce == 'mean':
            
                if f'{metric}_std' in reduced_data[j]['metric']:
                    if w is None:
                        raise ValueError("")
                    m = (w * reduced_data[j]['metric'][metric]).sum() / size
                    var_i = reduced_data[j]['metric'][f'{metric}_std']**2
                    m_i2 = reduced_data[j]['metric'][metric]**2
                    reduced_data[j]['metric'][f'{metric}_std'] = (((w*var_i).sum() + (w*m_i2).sum()) / size - m**2)**0.5
                    reduced_data[j]['metric'][metric] = m
                else:
                    reduced_data[j]['metric'][f'{metric}_std'] = reduced_data[j]['metric'][metric].std(axis=0)
                    reduced_data[j]['metric'][metric] = reduced_data[j]['metric'][metric].mean(axis=0)
            
            else:
                reduced_data[j]['metric'][metric] = getattr(np, reduce)(reduced_data[j]['metric'][metric], axis=0)
        
        if experiment_name is not None:
            reduced_data[j]['info'][experiment_name] = n

    return reduced_data

class Experiment:
    def __init__(self, copy_data=True):
        
        self._next = data_creator
        self._tail = self
        self._head = True
        
        self._parallel_next = None
        self._parallel_tail = self

        self.copy_data = copy_data
    
    def enqueue(self, experiment):
        self._tail._next = experiment
        self._tail = experiment._tail
        experiment._head = False
        return self
    
    def parallel(self, experiment, same_enqueue=True):
        self._parallel_tail._parallel_next = experiment
        self._parallel_tail = experiment._parallel_tail
        experiment._head = False

        if same_enqueue and self._next != data_creator:
            experiment.enqueue(self._next)
        
        return self

    @abstractclassmethod
    def _subclass_check_error(self, dataset, model):
        return dataset, model
        
    def __call__(self, datasets, models, **kwargs):
        
        data = []

        for dataset in self.__check_error_and_generate_datasets(datasets):
            for model in self.__check_error_and_generate_models(models):
                
                dataset, model = self._subclass_check_error(dataset, model)
                
                call_data = self.call(dataset, model, **kwargs)
                data.extend(call_data)
                
        if self._parallel_next is not None:
            parallel_data = self._parallel_next(datasets, models, **kwargs)
            data.extend(parallel_data)

        # joins info and metrics together
        if self._head:
            for i in range(len(data)):
                data[i]['info'].update(data[i]['metric'])
                data[i] = data[i]['info']
                
        return data

    def __check_error_and_generate_datasets(self, datasets):
        
        if type(datasets) not in (tuple, list, np.ndarray):
            datasets = [datasets]
        
        for dataset in datasets:

            has_preprocessings = False
            
            if type(dataset) in (tuple, list, np.ndarray):
                if len(dataset) != 2 :
                    raise ValueError("")
                dataset, preprocessings =  dataset
                has_preprocessings = True

            if not isinstance(dataset, Dataset):
                if not callable(dataset):
                    raise ValueError("")
                dataset_obj = dataset()
                if not isinstance(dataset_obj, Dataset):
                    raise ValueError("")
            else:
                dataset_obj = dataset

            if has_preprocessings:

                if not self.copy_data:
                    del dataset_obj
            
                if type(preprocessings) is not dict:
                    raise ValueError("")

                preprocessings_names = list(preprocessings.keys())
                
                # Make sure every value is an iterable in order to itertools.product to work
                for preprocessing_name in preprocessings_names:
                    if not hasattr(preprocessings[preprocessing_name], '__iter__'):
                        preprocessings[preprocessing_name] = [preprocessings[preprocessing_name]]
                
                for preprocessing_args in itertools.product(*preprocessings.values()):
                    
                    if self.copy_data:
                        iteration_dataset = copy.deepcopy(dataset_obj)
                    else:
                        iteration_dataset = dataset()
                    
                    for preprocessing_name, preprocessing_arg in zip(preprocessings_names, preprocessing_args):
                        
                        if preprocessing_arg is None:
                            getattr(iteration_dataset, preprocessing_name)()
                        else:
                            getattr(iteration_dataset, preprocessing_name)(preprocessing_arg)
                        
                    yield iteration_dataset

            else:
                yield dataset_obj
        
    def __check_error_and_generate_models(self, models):

        if type(models) not in (tuple, list, np.ndarray):
            models = [models]
        
        for model in models:

            has_hyperparemeters = False
            
            if type(model) in (tuple, list, np.ndarray):
                if len(model) != 2 :
                    raise ValueError("")
                model, hyperparemeters =  model
                has_hyperparemeters = True

                if not callable(model):
                    raise ValueError("")
            
            if has_hyperparemeters:
            
                if type(hyperparemeters) is not dict:
                    raise ValueError("")

                hyperparemeters_names = list(hyperparemeters.keys())
                
                # Make sure every value is an iterable in order to itertools.product to work
                for hyperparemeters_name in hyperparemeters_names:
                    if not hasattr(hyperparemeters[hyperparemeters_name], '__iter__'):
                        hyperparemeters[hyperparemeters_name] = [hyperparemeters[hyperparemeters_name]]
                
                for hyperparemeter_args in itertools.product(*hyperparemeters.values()):
                    
                    iteration_hyperparemeters = dict(zip(hyperparemeters_names, hyperparemeter_args))
                    iteration_model = lambda **kwargs : model(**iteration_hyperparemeters, **kwargs)

                yield iteration_model

            else:
                yield model


class DatasetExperiment(Experiment):
    def __init__(self, copy_data=True):
        super().__init__(copy_data)
    
    def _subclass_check_error(self, dataset, model):
        return dataset, model
        

class ModelExperiment(Experiment):
    def __init__(self):
        super().__init__()
    
    def _subclass_check_error(self, dataset, model):
        if not callable(model):
            raise ValueError("")
        return dataset, model


class StandartExperiment(Experiment):
    def __init__(self):
        super().__init__()

    def _subclass_check_error(self, dataset, model):
        if not isinstance(dataset, Dataset):
            dataset = dataset()
        if not isinstance(model, Model):
            model = model(num_inputs=dataset.x_dim[0], num_classes=dataset.y_dim)
        return dataset, model


class NullExperiment(Experiment):
    def __init__(self):
        super().__init__()
    
    def call(self, dataset, model):
        return self._next(dataset, model)


# def enqueue(*experiments):
    
#     head = NullExperiment()
    
#     for experiment in experiments:
        
#         if type(experiment) in (tuple, list):
            
#             if not isinstance(experiment[0], Experiment):
#                 raise ValueError("")
            
#             head.enqueue(experiment[0])
            
#             for parallel_experiment in experiment[1:]:
#                 if not isinstance(parallel_experiment, Experiment):
#                     raise ValueError("")
#                 experiment[0].parallel(parallel_experiment)
        
#         elif isinstance(experiment, Experiment):
#             head.enqueue(experiment)
        
#         else:
#             raise ValueError("")

#     return head


def enqueue(*experiments):
    
    head = NullExperiment()
    
    for experiment in experiments:
        
        if type(experiment) in (tuple, list):
            if not isinstance(experiment[0], Experiment):
                raise ValueError("")
            head.enqueue(experiment[0])
        
        elif isinstance(experiment, Experiment):
            head.enqueue(experiment)
        
        else:
            raise ValueError("")

    for experiment in reversed(experiments):

        if type(experiment) in (tuple, list):

            for parallel_experiment in experiment[1:]:
                if not isinstance(parallel_experiment, Experiment):
                    raise ValueError("")
                experiment[0].parallel(parallel_experiment)

    return head