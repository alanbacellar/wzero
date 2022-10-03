import numpy as np
from scipy.stats import norm, expon, exponnorm

def threshold(x, th):
    return x > th

def one_hot(x, lenght=None):
    if lenght == None:
        lenght = x.max()+1
    oh = np.zeros([x.size, lenght])
    oh[np.arange(x.size), x] = 1
    return oh

def thermometer(x, num_bits, min_, max_):
    t = (num_bits + 1) * (x.astype(np.float32) - min_) / (max_ - min_)
    return np.stack([t > i for i in range(1, num_bits+1)], axis=-1)

def bin_array(x, d):
    return np.fromstring(np.binary_repr(x, width=d), np.int8) - 48

def gaussian_thresholds(x,  num_bits, individual=True):
    std_skews = norm.ppf(np.arange(1, num_bits+1) / (num_bits+1))
    mean = x.mean(axis=0) if individual else x.mean()
    std = x.std(axis=0) if individual else x.std() 
    thresholds = np.array([std_skew * std + mean for std_skew in std_skews])
    return thresholds

def gaussian_thermometer(x, num_bits=1, thresholds=None, individual=True):
    if thresholds is None:
        thresholds = gaussian_thresholds(x, num_bits, individual=individual)
    terms = [x > threshold for threshold in thresholds]
    return np.stack(terms, axis=-1)

def exponential_thresholds(x, num_bits, individual=True):
    std_skews = expon.ppf(np.arange(1, num_bits+1) / (num_bits+1))
    if individual:
        mean = np.zeros(x.shape[1])
        std = np.zeros(x.shape[1])
        for i in range(x.shape[1]):
            mean[i], std[i] = expon.fit(x[:, i])
    else:
        mean, std = expon.fit(x)
    thresholds = np.array([std_skew * std + mean for std_skew in std_skews])
    return thresholds

def exponential_thermometer(x, num_bits=1, thresholds=None, individual=True):
    if thresholds is None:
        thresholds = exponential_thresholds(x, num_bits, individual=individual)
    terms = [x > threshold for threshold in thresholds]
    return np.stack(terms, axis=-1)

def expnorm_thresholds(x,  num_bits, individual=True):
    if individual:
        thresholds = np.zeros([num_bits, x.shape[1]])
        for i in range(x.shape[1]):
            K, mean, std = exponnorm.fit(x[:, i])
            std_skews = exponnorm.ppf(np.arange(1, num_bits+1) / (num_bits+1), K)
            thresholds[:, i] = np.array([std_skew * std + mean for std_skew in std_skews])
    else:
        K, mean, std = exponnorm.fit(x)
        std_skews = exponnorm.ppf(np.arange(1, num_bits+1) / (num_bits+1), K)
        thresholds = np.array([std_skew * std + mean for std_skew in std_skews])
    return thresholds

def expnorm_thermometer(x, num_bits=1, thresholds=None, individual=True):
    if thresholds is None:
        thresholds = expnorm_thresholds(x, num_bits, individual=individual)
    terms = [x > threshold for threshold in thresholds]
    return np.stack(terms, axis=-1)

def get_splits(x, num_bits=1, individual=True):
    data = np.sort(x.flatten()) if not individual else np.sort(x, axis=0)
    indicies = np.array([int(data.shape[0]*i/(num_bits+1)) for i in range(1, num_bits+1)])
    return data[indicies]

# def get_splits(x, num_bits=1, individual=True):
#     data = np.sort(x.flatten()) if not individual else np.sort(x, axis=0)
#     if individual:
#         thresholds = []
#         for i in range(data.shape[1]):
#             data_unq_i = np.unique(data[:, i])
#             indicies = [int(data_unq_i.shape[0]*i/(num_bits+1)) for i in range(1, num_bits+1)]
#             thresholds.append(data_unq_i[indicies])
#         thresholds = np.array(thresholds)
#         return thresholds.T
#     else:
#         data_unq = np.unique(data)
#         indicies = [int(data_unq.shape[0]*i/(num_bits+1)) for i in range(1, num_bits+1)]
#         return data[indicies]
    
def distrib_therm(x, num_bits=None, splits=None, individual=True):
    if splits is None:
        splits = get_splits(x, num_bits, individual=individual)
    return np.stack([x > split for split in splits], axis=-1)

def get_splits2(x, y, num_bits=1):
    indicies = np.argsort(x, axis=0)
    thresholds = []
    for i in range(x.shape[1]):
        data = x[:, i][indicies[:, i]]
        labels = y[indicies[:, i]]
        label_index = []
        last_label = labels[0]
        for j, label in enumerate(labels[1:]):
            if label == last_label:
                continue
            last_label = label
            label_index.append(j+1)
        label_index = np.array(label_index)
        indexs = [int(label_index.shape[0]*i/(num_bits+1)) for i in range(1, num_bits+1)]
        indexs = label_index[indexs]
        thresholds.append(data[indexs])
    thresholds = np.array(thresholds)
    return thresholds.T

def distrib_therm2(x, splits):
    return np.stack([x > split for split in splits], axis=-1)