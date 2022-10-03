import wzero as wz
import wnnc
import wisardpkg as wp
import numpy as np
from wzero import utils
from wzero import metrics

def mean_profile(model_gen, dataset, n=1, raw=False, predict_args=[]):
    
    data = {
        'train_time': np.empty(n),
        'test_time': np.empty(n),
        'accuracy': np.empty(n)
    }

    for i in range(n):

        model = model_gen()

        data['train_time'][i], null = utils.timeit(model.train, dataset.x_train, dataset.y_train)
        try:
            data['test_time'][i], predictions = utils.timeit(model.predict, dataset.x_test, *predict_args)
        except:
            data['test_time'][i], predictions = utils.timeit(model.classify, dataset.x_test, *predict_args)
        try:
            data['accuracy'][i] = metrics.accuracy(dataset.y_test, predictions)
        except:
            #print(np.array(dataset.y_test).shape)
            #print(np.array(predictions).shape)
            data['accuracy'][i] = (np.array(dataset.y_test).astype(str) == np.array(predictions).astype(str)).sum() / len(dataset.y_test)
    if raw:
        return data
    
    data['train_time'], data['train_time_std'] = data['train_time'].mean(), data['train_time'].std()
    data['test_time'], data['test_time_std'] = data['test_time'].mean(), data['test_time'].std()
    data['accuracy'], data['accuracy_std'] = data['accuracy'].mean(), data['accuracy'].std()

    return data

dataset = wz.datasets.FashionMnist().thermometer(10).flatten()

# dataset2 = wz.datasets.FashionMnist().thermometer(10).flatten()
# dataset2.x_train = dataset.x_train.astype(int).tolist()
# dataset2.y_train = dataset.y_train.astype(str).tolist()
# dataset2.x_test = dataset.x_test.astype(int).tolist()
# dataset2.y_test = dataset.y_test.astype(str).tolist()


# ww = lambda : wp.Wisard(40)
wi = lambda : wz.models.WiSARD(dataset.x_dim[0], 40, dataset.y_dim)
wi2 = lambda : wnnc.CcWiSARD2(dataset.x_dim[0], 40, dataset.y_dim)
wi3 = lambda : wnnc.CcWiSARD3(dataset.x_dim[0], 40, dataset.y_dim)

# print('Wp')
# dataw = mean_profile(ww, dataset2, n=10)
print('Wi')
data = mean_profile(wi, dataset, n=5)
print('Wi2')
data2 = mean_profile(wi2, dataset, n=5)
print('Wi3')
data3 = mean_profile(wi3, dataset, n=5)

# print(dataw)
print(data)
print(data2)
print(data3)