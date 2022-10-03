import numpy as np

from .. import utils
from .experiment import StandartExperiment, metrics_dict

class Train(StandartExperiment):
    def __init__(self, metrics=[]):
        super().__init__()
        
        if type(metrics) is not list:
            raise ValueError("")
        
        self.metrics = metrics

        self.benchmark = 'time' in metrics
        if self.benchmark:
            self.metrics.remove('time')
        
        for i in range(len(self.metrics)):
            if type(self.metrics[i]) is str:
                self.metrics[i] = metrics_dict[self.metrics[i]]

    def call(self, dataset, model):

        train_time, null = utils.timeit(model.train, dataset.x_train, dataset.y_train)

        datas = self._next(dataset, model)

        if len(self.metrics):
            train_pred_time, predictions = utils.timeit(model.predict, dataset.x_train)

        for data in datas:

            for metric in self.metrics:
                data['metric'][f'train_{metric.__name__}'] = metric(dataset.y_train, predictions)

            if self.benchmark:
                data['metric']['train_time'] = train_time
                if len(self.metrics):
                    data['metric']['train_pred_time'] = train_pred_time

        return datas


class Test(StandartExperiment):
    def __init__(self, metrics=['accuracy']):
        super().__init__()

        if type(metrics) is not list:
            raise ValueError("")
        
        self.metrics = metrics

        self.benchmark = 'time' in metrics
        if self.benchmark:
            self.metrics.remove('time')
        
        for i in range(len(self.metrics)):
            if type(self.metrics[i]) is str:
                self.metrics[i] = metrics_dict[self.metrics[i]]


    def call(self, dataset, model):

        datas = self._next(dataset, model)
        
        if len(self.metrics):
            test_time, predictions = utils.timeit(model.predict, dataset.x_test)

        for data in datas:

            for metric in self.metrics:
                data['metric'][f'test_{metric.__name__}'] = metric(dataset.y_test, predictions)

            if self.benchmark:
                data['metric']['test_time'] = test_time

        return datas


def TrainAndTest(benchmark=False):
    return Train(benchmark).enqueue(Test(benchmark))