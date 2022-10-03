# from abc import abstractclassmethod
# import numpy as np
# import functools
# import types
# import itertools
# import copy

# from . import metrics
# from . import utils
# from .datasets.dataset import Dataset
# from .models.model import Model

# def data_creator(dataset, model):
    
#     data = [{
#         'info': {
#             'dataset': dataset.name,
#             'pre_processings': dataset.pre_processings,
#             'model':  model.__class__.__name__.lower(),
#             'hyper_parameters': model.hyperparameters
#         },
#         'metric': {}
#     }]
    
#     return data

# metrics_dict = {
#     'accuracy': metrics.accuracy
# }

# def reduce_datas(reduce, datas, w=None):

#     n = len(datas)

#     reduced_data = datas[0]

#     if w is not None:
#         if type(w) is not np.ndarray:
#             raise ValueError("")
#         size = w.sum()

#     for i in range(n):
#         for j in range(len(datas[0])):
#             for metric in datas[0][0]['metric']:
                
#                 if i == 0 and j == 0:
#                     tmp = reduced_data[j]['metric'][metric]
#                     if type(tmp) is not np.ndarray:
#                         reduced_data[j]['metric'][metric] = np.zeros(n)
#                     else:
#                         reduced_data[j]['metric'][metric] = np.zeros([n, *tmp.shape])
#                     reduced_data[j]['metric'][metric][0] = tmp
                
#                 else:
#                     reduced_data[j]['metric'][metric][i] = datas[i][j]['metric'][metric]

#     for j in range(len(datas[0])):
#         for metric in list(datas[0][0]['metric'].keys()):
            
#             if 'std' in metric:
#                 continue
            
#             if reduce == 'mean':
            
#                 if f'{metric}_std' in reduced_data[j]['metric']:
#                     if w is None:
#                         raise ValueError("")
#                     m = (w * reduced_data[j]['metric'][metric]).sum() / size
#                     var_i = reduced_data[j]['metric'][f'{metric}_std']**2
#                     m_i2 = reduced_data[j]['metric'][metric]**2
#                     reduced_data[j]['metric'][f'{metric}_std'] = (((w*var_i).sum() + (w*m_i2).sum()) / size - m**2)**0.5
#                     reduced_data[j]['metric'][metric] = m
#                 else:
#                     reduced_data[j]['metric'][f'{metric}_std'] = reduced_data[j]['metric'][metric].std(axis=0)
#                     reduced_data[j]['metric'][metric] = reduced_data[j]['metric'][metric].mean(axis=0)
            
#             else:
#                 reduced_data[j]['metric'][metric] = getattr(np, reduce)(reduced_data[j]['metric'][metric], axis=0)
        
        
#         reduced_data[j]['info']['kfold'] = n

#     return reduced_data

# class Experiment:
#     def __init__(self, copy_data=True):
        
#         self._next = data_creator
#         self._tail = self
        
#         self._parallel_next = None
#         self._parallel_tail = self

#         self.copy_data = copy_data
        
#     def __call_func(self, dataset, model, **kwargs):
#         data = self.call(dataset, model, **kwargs)
#         if self._parallel_next is not None:
#             parallel_data = self._parallel_next(dataset, model, **kwargs)
#             data.extend(parallel_data)
#         return data

#     def __call__(self, datasets, models, **kwargs):
        
#         data = []
        
#         if not isinstance(self, StandartExperiment):
#             for dataset in self.generate_datasets(datasets):
#                 for model in self.generate_models(models):
#                     data.extend(self.__call_func(dataset, model))
#         else:
#             data.extend(self.__call_func(datasets, models))
        
#         return data

#     def generate_datasets(self, dataset):
        
#         if type(dataset) not in (tuple, list, np.ndarray):
#             dataset = [dataset]
#         datasets = dataset
        
#         for dataset in datasets:

#             has_preprocessings = False
            
#             if type(dataset) in (tuple, list, np.ndarray):
#                 if len(dataset) != 2 :
#                     raise ValueError("")
#                 dataset, preprocessings =  dataset
#                 has_preprocessings = True

#             if not isinstance(dataset, Dataset):
#                 if not callable(dataset):
#                     raise ValueError("")
#                 dataset_obj = dataset()
#                 if not isinstance(dataset_obj, Dataset):
#                     raise ValueError("")
#             else:
#                 dataset_obj = dataset

#             if has_preprocessings:

#                 if not self.copy_data:
#                     del dataset_obj
            
#                 if type(preprocessings) is not dict:
#                     raise ValueError("")

#                 preprocessings_names = list(preprocessings.keys())
                
#                 # Make sure every value is an iterable in order to itertools.product to work
#                 for preprocessing_name in preprocessings_names:
#                     if not hasattr(preprocessings[preprocessing_name], '__iter__'):
#                         preprocessings[preprocessing_name] = [preprocessings[preprocessing_name]]
                
#                 for preprocessing_args in itertools.product(*preprocessings.values()):
                    
#                     if self.copy_data:
#                         iteration_dataset = copy.deepcopy(dataset_obj)
#                     else:
#                         iteration_dataset = dataset()
                    
#                     for preprocessing_name, preprocessing_arg in zip(preprocessings_names, preprocessing_args):
                        
#                         if preprocessing_arg is None:
#                             getattr(iteration_dataset, preprocessing_name)()
#                         else:
#                             getattr(iteration_dataset, preprocessing_name)(preprocessing_arg)
                        
#                     yield iteration_dataset

#             else:
#                 yield dataset_obj
        
#     def generate_models(self, model):

#         if type(model) not in (tuple, list, np.ndarray):
#             model = [model]
#         models = model
        
#         for model in models:

#             has_hyperparemeters = False
            
#             if type(model) in (tuple, list, np.ndarray):
#                 if len(model) != 2 :
#                     raise ValueError("")
#                 model, hyperparemeters =  model
#                 has_hyperparemeters = True

#                 if not callable(model):
#                     raise ValueError("")
            
#             if has_hyperparemeters:
            
#                 if type(hyperparemeters) is not dict:
#                     raise ValueError("")

#                 hyperparemeters_names = list(hyperparemeters.keys())
                
#                 # Make sure every value is an iterable in order to itertools.product to work
#                 for hyperparemeters_name in hyperparemeters_names:
#                     if not hasattr(hyperparemeters[hyperparemeters_name], '__iter__'):
#                         hyperparemeters[hyperparemeters_name] = [hyperparemeters[hyperparemeters_name]]
                
#                 for hyperparemeter_args in itertools.product(*hyperparemeters.values()):
                    
#                     iteration_hyperparemeters = dict(zip(hyperparemeters_names, hyperparemeter_args))
#                     iteration_model = lambda **kwargs : model(**iteration_hyperparemeters, **kwargs)

#                 yield iteration_model

#             else:
#                 yield model


#     def enqueue(self, experiment):
#         self._tail._next = experiment
#         self._tail = experiment._tail
    
#     def parallel(self, experiment, same_enqueue=True):
#         self._parallel_tail._parallel_next = experiment
#         self._parallel_tail = experiment._parallel_tail

#         if same_enqueue:
#             experiment.enqueue(self._next)


# class DatasetExperiment(Experiment):
#     def __init__(self, copy_data=True):
#         self.copy_data = copy_data
#         super().__init__()
        
#     def __call__(self, dataset, model, **kwargs):
        
#         if type(dataset) not in (tuple, list, np.ndarray):
#             dataset = [dataset]
#         datasets = dataset
        
#         data = []
        
#         for dataset in datasets:

#             has_preprocessings = False
            
#             if type(dataset) in (tuple, list, np.ndarray):
#                 if len(dataset) != 2 :
#                     raise ValueError("")
#                 dataset, preprocessings =  dataset
#                 has_preprocessings = True

#             if not isinstance(dataset, Dataset):
#                 if not callable(dataset):
#                     raise ValueError("")
#                 dataset_obj = dataset()
#                 if not isinstance(dataset_obj, Dataset):
#                     raise ValueError("")
#             else:
#                 dataset_obj = dataset

#             if has_preprocessings:

#                 if not self.copy_data:
#                     del dataset_obj
            
#                 if type(preprocessings) is not dict:
#                     raise ValueError("")

#                 preprocessings_names = list(preprocessings.keys())
                
#                 # Make sure every value is an iterable in order to itertools.product to work
#                 for preprocessing_name in preprocessings_names:
#                     if not hasattr(preprocessings[preprocessing_name], '__iter__'):
#                         preprocessings[preprocessing_name] = [preprocessings[preprocessing_name]]
                
#                 for preprocessing_args in itertools.product(*preprocessings.values()):
                    
#                     if self.copy_data:
#                         iteration_dataset = copy.deepcopy(dataset_obj)
#                     else:
#                         iteration_dataset = dataset()
                    
#                     for preprocessing_name, preprocessing_arg in zip(preprocessings_names, preprocessing_args):
                        
#                         if preprocessing_arg is None:
#                             getattr(iteration_dataset, preprocessing_name)()
#                         else:
#                             getattr(iteration_dataset, preprocessing_name)(preprocessing_arg)
                        
#                         call_data = super().__call__(iteration_dataset, model, **kwargs)
#                         data.extend(call_data)

#             else:
#                 call_data = super().__call__(dataset_obj, model, **kwargs)
#                 data.extend(call_data)
        
#         return data


# class ModelExperiment(Experiment):
#     def __init__(self):
#         super().__init__()
    
#     def __call__(self, dataset, model, **kwargs):

#         if type(model) not in (tuple, list, np.ndarray):
#             model = [model]
#         models = model
        
#         data = []
        
#         for model in models:

#             has_hyperparemeters = False
            
#             if type(model) in (tuple, list, np.ndarray):
#                 if len(model) != 2 :
#                     raise ValueError("")
#                 model, hyperparemeters =  model
#                 has_hyperparemeters = True

#             if not callable(model):
#                 raise ValueError("")
            
#             if has_hyperparemeters:
            
#                 if type(hyperparemeters) is not dict:
#                     raise ValueError("")

#                 hyperparemeters_names = list(hyperparemeters.keys())
                
#                 # Make sure every value is an iterable in order to itertools.product to work
#                 for hyperparemeters_name in hyperparemeters_names:
#                     if not hasattr(hyperparemeters[hyperparemeters_name], '__iter__'):
#                         hyperparemeters[hyperparemeters_name] = [hyperparemeters[hyperparemeters_name]]
                
#                 for hyperparemeter_args in itertools.product(*hyperparemeters.values()):
                    
#                     iteration_hyperparemeters = dict(zip(hyperparemeters_names, hyperparemeter_args))
#                     iteration_model = lambda **kwargs : model(**iteration_hyperparemeters, **kwargs)

#                     call_data = super().__call__(dataset, iteration_model, **kwargs)
#                     data.extend(call_data)

#             else:
#                 call_data = super().__call__(dataset, model, **kwargs)
#                 data.extend(call_data)
        
#         return data


# class StandartExperiment(Experiment):
#     def __init__(self):
#         super().__init__()
        
#     def __call__(self, dataset, model, **kwargs):
        
#         if not isinstance(dataset, Dataset):
#             dataset = dataset()
        
#         if not isinstance(model, Model):
#             model = model(num_inputs=dataset.x_dim[0], num_classes=dataset.y_dim)
        
#         data = super().__call__(dataset, model, **kwargs)
#         return data


# class Train(StandartExperiment):
#     def __init__(self, metrics=[]):
#         super().__init__()
        
#         if type(metrics) is not list:
#             raise ValueError("")
        
#         self.metrics = metrics

#         self.benchmark = 'time' in metrics
#         if self.benchmark:
#             self.metrics.remove('time')
        
#         for i in range(len(self.metrics)):
#             if type(self.metrics[i]) is str:
#                 self.metrics[i] = metrics_dict[self.metrics[i]]

#     def call(self, dataset, model):

#         train_time, null = utils.timeit(model.train, dataset.x_train, dataset.y_train)

#         datas = self._next(dataset, model)

#         if len(self.metrics):
#             train_pred_time, predictions = utils.timeit(model.predict, dataset.x_train)

#         for data in datas:

#             for metric in self.metrics:
#                 data['metric'][f'train_{metric.__name__}'] = metric(dataset.y_train, predictions)

#             if self.benchmark:
#                 data['metric']['train_time'] = train_time
#                 if len(self.metrics):
#                     data['metric']['train_pred_time'] = train_pred_time

#         return datas


# class Test(StandartExperiment):
#     def __init__(self, metrics=['accuracy']):
#         super().__init__()

#         if type(metrics) is not list:
#             raise ValueError("")
        
#         self.metrics = metrics

#         self.benchmark = 'time' in metrics
#         if self.benchmark:
#             self.metrics.remove('time')
        
#         for i in range(len(self.metrics)):
#             if type(self.metrics[i]) is str:
#                 self.metrics[i] = metrics_dict[self.metrics[i]]


#     def call(self, dataset, model):

#         datas = self._next(dataset, model)
        
#         if len(self.metrics):
#             test_time, predictions = utils.timeit(model.predict, dataset.x_test)

#         for data in datas:

#             for metric in self.metrics:
#                 data['metric'][f'test_{metric.__name__}'] = metric(dataset.y_test, predictions)

#             if self.benchmark:
#                 data['metric']['test_time'] = test_time

#         return datas


# def TrainAndTest(benchmark=False):
#     return Train(benchmark).enqueue(Test(benchmark))


# class Mappings(ModelExperiment):
#     def __init__(self, num_mappings=10, reduce='mean'):
#         super().__init__()

#         self.num_mappings = num_mappings

#         if reduce not in (None, 'mean', 'median', 'max', 'min'):
#             raise ValueError("")

#         self.reduce = reduce

#     def call(self, dataset, model):
        
#         mappings_datas = [self._next(dataset, model) for i in range(self.num_mappings)]
        
#         return reduce_datas(self.reduce, mappings_datas)


# class KFold(DatasetExperiment):
#     def __init__(self, k=5, reduce='mean', shuffle=True, seed=None, skip_splitted=False):
#         super().__init__()

#         if type(k) is not int:
#             raise ValueError("")
#         if k <= 0:
#             raise ValueError("")
#         self.k = k

#         if reduce not in (None, 'mean', 'median', 'max', 'min'):
#             raise ValueError("")
#         self.reduce = reduce

#         if type(shuffle) is not bool:
#             raise ValueError("")
#         self.shuffle = shuffle

#         if not(type(seed) is int or seed is None):
#             raise ValueError("")
#         self.seed = seed

#         if type(skip_splitted) is not bool:
#             raise ValueError("")
#         self.skip_splitted = skip_splitted

    
#     def call(self, dataset, model):

#         if dataset.splitted:
#             if self.skip_splitted:
#                 return []
#             else:
#                 dataset.join()
        
#         if self.shuffle:
#             if self.seed is not None:
#                 dataset.shuffle(self.seed)
#             else:
#                 dataset.shuffle()

#         fold_dataset = copy.copy(dataset)
        
#         size = fold_dataset.features.shape[0]
#         k_size = size // self.k
#         k_mod = size % self.k

#         fold_dataset.splitted = True
#         del fold_dataset.features
#         del fold_dataset.labels

#         w = np.array([k_size + int(i < k_mod) for i in range(self.k)])

#         fold_datas = []

#         begin = 0
#         for i in range(self.k):
            
#             end = begin + w[i]
            
#             fold_dataset.x_train = dataset.features[begin:end]
#             fold_dataset.y_train = dataset.labels[begin:end]

#             fold_dataset.x_test = np.concatenate([dataset.features[:begin], dataset.features[end:]], axis=0)
#             fold_dataset.y_test = np.concatenate([dataset.labels[:begin], dataset.labels[end:]], axis=0)

#             begin = end

#             fold_datas.append(self._next(fold_dataset, model))

#         return reduce_datas(self.reduce, fold_datas, w)


# class NullExperiment(Experiment):
#     def __init__(self):
#         super().__init__()
    
#     def call(self, dataset, model):
#         return self._next(dataset, model)


# def stack(*experiments):
    
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


# # my_experiment = stack(
# #     Train(),
# #     Test()
# # )

# # results = my_experiment(WiSARD(7840, 40, 10), Mnist().thermometer(10).flatten())


# # my_experiment = stack(
# #     KFold(10),
# #     Hyperparemeters(),
# #     Mappings(100), 
# #     Train(metrics=['time']),
# #     Test(metrics=['time', 'accuracy'])
# # )

# # results = my_experiment(WiSARD, Mnist())

# # datasets = (
# #     Ecoli().thermometer(20).flatten(),
# #     Segment().thermometer(20).flatten(),
# #     SatImage().thermometer(20).flatten(),
# #     Shuttle().thermometer(20).flatten(),
# #     Mnist().thermometer(10).flatten(),
# #     FashionMnist().thermometer(10).flatten()
# # )

# # models = (
# #     lambda num_inputs, num_classes : WiSARD(num_inputs, 40, num_classes),
# #     lambda num_inputs, num_classes : BleachingWiSARD(num_inputs, 40, num_classes)
# # )

# # datasets = (
# #     ( Ecoli, {'thermoemter': range(5, 20), 'flatten': None} ),
# #     ( Segment, {'thermoemter': range(5, 20), 'flatten': None} ),
# #     ( SatImage, {'thermoemter': range(5, 20), 'flatten': None} ),
# #     ( Shuttle, {'thermoemter': range(5, 20), 'flatten': None} ),
# #     ( Mnist, {'thermoemter': range(5, 10), 'flatten': None} ),
# #     ( FashionMnist, {'thermoemter': range(5, 10), 'flatten': None} )
# # )

# # models = (
# #     ( WiSARD, {'n': range(1, 64)} ),
# #     ( BleachingWiSARD, {'n': range(1, 64)} ),
# #     ( BloomWiSARD, {'m': range(1, 64), 'n': range(1, 40), 'b': 13} ),
# # )

# # datasets = (
# #     (
# #         Ecoli, 
# #         {'thermoemter': range(5, 20),
# #          'flatten': None,
# #          'train_test_split': (0.5, 0.6, 0.7, 0.8, 0.9)} 
# #     ),
# #     (
# #         Segment, 
# #         {'thermoemter': range(5, 20)} 
# #     ),
# #     (
# #         SatImage, 
# #         {'thermoemter': range(5, 20)} 
# #     ),
# #     (
# #         Shuttle, 
# #         {'thermoemter': range(5, 20)} 
# #     ),
# #     (
# #         Mnist, 
# #         {'thermoemter': range(5, 10)} 
# #     ),
# #     (
# #         FashionMnist, 
# #         {'thermoemter': range(5, 10)} 
# #     )
# # )

# # models = (
# #     ( 
# #         WiSARD, 
# #         {'n': range(1, 64)} 
# #     ),
# #     ( 
# #         BleachingWiSARD, 
# #         {'n': range(1, 64)} 
# #     ),
# #     ( 
# #         BloomWiSARD, 
# #         {'m': range(1, 64), 
# #          'n': range(1, 40), 
# #          'b': 13} 
# #     ),
# # )



# # results = my_experiment(models, datasets, hyperparemeters)


# def profile2(dataset, model, hyperparemeters, predict_args=[]):
    
#     data = {}

#     data['train_time'], null = utils.timeit(model.train, dataset.x_train, dataset.y_train)
#     data['pred_train_time'], predictions = utils.timeit(model.predict, dataset.x_train, *predict_args)
#     data['train_acc'] = metrics.accuracy(dataset.y_train, predictions)
#     data['pred_test_time'], predictions = utils.timeit(model.predict, dataset.x_test, *predict_args)
#     data['test_acc'] = metrics.accuracy(dataset.y_test, predictions)
    
#     return data

import numpy as np
from . import utils
from . import metrics

def profile(dataset, model, predict_args=[]):
    
    data = {}
    data['train_time'], null = utils.timeit(model.train, dataset.x_train, dataset.y_train)
    data['test_time'], predictions = utils.timeit(model.predict, dataset.x_test, *predict_args)
    data['accuracy'] = metrics.accuracy(dataset.y_test, predictions)
    
    return data

def mean_profile(dataset, model_gen, n=1, raw=False, predict_args=[]):
    
    data = {
        'train_time': np.empty(n),
        'test_time': np.empty(n),
        'accuracy': np.empty(n)
    }

    for i in range(n):

        model = model_gen()

        data['train_time'][i], null = utils.timeit(model.train, dataset.x_train, dataset.y_train)
        data['test_time'][i], predictions = utils.timeit(model.predict, dataset.x_test, *predict_args)
        data['accuracy'][i] = metrics.accuracy(dataset.y_test, predictions)
    
    if raw:
        return data
    
    data['train_time'], data['train_time_std'] = data['train_time'].mean(), data['train_time'].std()
    data['test_time'], data['test_time_std'] = data['test_time'].mean(), data['test_time'].std()
    data['accuracy'], data['accuracy_std'] = data['accuracy'].mean(), data['accuracy'].std()

    return data
