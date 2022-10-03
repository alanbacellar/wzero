from .wisard import WiSARD
from .bleaching_wisard import BleachingWiSARD
from .bloom_wisard import BloomWiSARD
from .bleaching_bloom_wisard import BleachingBloomWiSARD
from .regression_wisard import RegressionWiSARD
from .model import Model
from .wisard2 import WiSARD2
from .bleaching_wisard2 import BleachingWiSARD2
from .wisardpc1D import WiSARDPC1D
from .wisardpc2D import WiSARDPC2D
from .bloom_ram_wisard import BloomRamWiSARD
from .bleaching_wisard15 import BleachingWiSARD15
from .bloom_wisard2 import BloomWiSARD2
from .bloom_wisard3 import BloomWiSARD3
from .bloom_wisard4 import BloomWiSARD4
from .bloom_wisard5 import BloomWiSARD5
from .bloom_wisard55 import BloomWiSARD55
from .prob_wisard import ProbWiSARD

import numpy as np


import random

class ConvWiSARD:
    def __init__(self, kernel_size, stride, n_bits, n_out, canonical=False, termometer=False):
        self.kernel_size = kernel_size
        self.stride = stride
        self.n_in = kernel_size ** 2
        if termometer:
            self.n_in *= termometer
        self.n_bits = n_bits
        self.n_out = n_out
        self.canonical = canonical
        self.__model = WiSARD2(self.n_in, n_bits, n_out)
    
    def get_x(self, x):

        X = []

        for i in range(0, x.shape[1], self.stride):
            for j in range(0, x.shape[2], self.stride):
                
                if i + self.kernel_size > x.shape[1]:
                    continue
                if j + self.kernel_size > x.shape[2]:
                    continue 
                
                X.append(x[:, i:i+self.kernel_size, j:j+self.kernel_size])

        # print(X, len(X))
        X = np.concatenate(X, axis=0)
        # print(X.shape)
        X = X.reshape(X.shape[0], -1)

        # print(X.shape)

        return X

    def train(self, x, y):

        X = self.get_x(x)
        Y = np.concatenate([y]*(X.shape[0]//x.shape[0]), axis=0)
        n = X.shape[0] // x.shape[0]
        print(X.shape)
        for i in range(n):
            # print(i)
            self.__model.train(X[i*x.shape[0]:(i+1)*x.shape[0]], Y[i*x.shape[0]:(i+1)*x.shape[0]])

        # self.__model.train(X, Y)

    def predict(self, x):

        X = self.get_x(x)
        
        out = np.zeros([x.shape[0], self.n_out])

        n = X.shape[0] // x.shape[0]
        for i in range(n):
            # out = np.maximum(out, pred[i*x.shape[0]:(i+1)*x.shape[0]])
            # out += pred[i*x.shape[0]:(i+1)*x.shape[0]]
            pred = self.__model.predict(X[i*x.shape[0]:(i+1)*x.shape[0]])
            out += pred

        return out
    
    # def predict(self, x):

    #     X = self.get_x(x)

    #     n = X.shape[0] // x.shape[0]
        
    #     out = np.zeros([n, x.shape[0], self.n_out])
        
    #     for i in range(n):
    #         # out = np.maximum(out, pred[i*x.shape[0]:(i+1)*x.shape[0]])
    #         # out += pred[i*x.shape[0]:(i+1)*x.shape[0]]
    #         pred = self.__model.predict(X[i*x.shape[0]:(i+1)*x.shape[0]])
    #         out[i] = pred

    #     out_sorted = np.sort(out, axis=-1)
        
    #     confidence = out_sorted[:,:,-1] - out_sorted[:,:,-2]
        
    #     k = 1

    #     output = np.zeros([x.shape[0], self.n_out])
        
    #     for kk in range(k):

    #         top_confidence = np.argmax(confidence, axis=0)
    #         for i, top in enumerate(top_confidence):
    #             output[i] += out[top, i] # ** confidence[top, i]
    #             # print(out[top, i])
    #         confidence[top][i] = 0

    #     return output
    
    def fit(self, x, y, batch_size, learning_rate):
        X = self.get_x(x)
        Y = np.concatenate([y]*(X.shape[0]//x.shape[0]), axis=0)
        n = X.shape[0] // x.shape[0]
        for i in range(n):
            self.__model.fit2(X[i*x.shape[0]:(i+1)*x.shape[0]], Y[i*x.shape[0]:(i+1)*x.shape[0]], batch_size, learning_rate)
    
    def clear(self):
        self.__model.clear()

    def mental_images(self):
        return self.__model.mental_images()

# class ConvBloomWiSARD:
#     def __init__(self, kernel_size, stride, n_bits, n_out, num_filters, filter_tuple_lenght, canonical=False, termometer=False):
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.n_in = kernel_size ** 2
#         if termometer:
#             self.n_in *= termometer
#         self.n_bits = n_bits
#         self.n_out = n_out
#         self.canonical = canonical
#         self.__model = wnnc.BloomWiSARD(self.n_in, n_bits, n_out, canonical, num_filters, filter_tuple_lenght)
    
#     def get_x(self, x):

#         X = []

#         for i in range(0, x.shape[1], self.stride):
#             for j in range(0, x.shape[2], self.stride):
                
#                 if i + self.kernel_size > x.shape[1]:
#                     continue
#                 if j + self.kernel_size > x.shape[2]:
#                     continue
                
#                 X.append(x[:, i:i+self.kernel_size, j:j+self.kernel_size])

#         # print(X, len(X))
#         X = np.concatenate(X, axis=0)
#         # print(X.shape)
#         X = X.reshape(X.shape[0], -1)

#         # print(X.shape)

#         return X

#     def train(self, x, y):

#         X = self.get_x(x)
#         Y = np.concatenate([y]*(X.shape[0]//x.shape[0]), axis=0)
#         n = X.shape[0] // x.shape[0]
#         print(X.shape)
#         for i in range(n):
#             # print(i)
#             self.__model.train(X[i*x.shape[0]:(i+1)*x.shape[0]], Y[i*x.shape[0]:(i+1)*x.shape[0]])

#         # self.__model.train(X, Y)

#     def predict(self, x):

#         X = self.get_x(x)
        
#         out = np.zeros([x.shape[0], self.n_out])

#         n = X.shape[0] // x.shape[0]
#         for i in range(n):
#             # out = np.maximum(out, pred[i*x.shape[0]:(i+1)*x.shape[0]])
#             # out += pred[i*x.shape[0]:(i+1)*x.shape[0]]
#             pred = self.__model.predict(X[i*x.shape[0]:(i+1)*x.shape[0]])
#             out += pred

#         return out
    
#     def fit(self, x, y, batch_size, learning_rate):
#         X = self.get_x(x)
#         Y = np.concatenate([y]*(X.shape[0]//x.shape[0]), axis=0)
#         n = X.shape[0] // x.shape[0]
#         for i in range(n):
#             self.__model.fit2(X[i*x.shape[0]:(i+1)*x.shape[0]], Y[i*x.shape[0]:(i+1)*x.shape[0]], batch_size, learning_rate)
    
#     def clear(self):
#         self.__model.clear()

#     def mental_images(self):
#         return self.__model.mental_images()


# # from .cpp import wnnc
# # import numpy as np

# # class WiSARD:
# #     def __init__(self, n_in, n_bits, n_out, canonical=False):
# #         self.n_in = n_in
# #         self.n_bits = n_bits
# #         self.n_out = n_out
# #         self.canonical = canonical
# #         self.__model = wnnc.WiSARD(n_in, n_bits, n_out, canonical)
    
# #     def train(self, x, y):
# #         if type(x) == list:
# #             x = np.array(x)
# #         if type(y) == list:
# #             y = np.array(y)
# #         self.__model.train(x.astype(bool), y.astype(np.int32))
    
# #     def predict(self, x):
# #         if type(x) == list:
# #             x = np.array(x)
# #         self.__model.predict(x.astype(bool))
    
# #     def clear(self):
# #         self.__model.clear()


# # class HybridWiSARD:
# #     def __init__(self, n_in, n_bits, n_out, canonical=False):
# #         self.n_in = n_in
# #         self.n_bits = n_bits
# #         self.n_out = n_out
# #         self.canonical = canonical
# #         self.__model = wnnc.HybridWiSARD(n_in, n_bits, n_out, canonical)
    
# #     def train(self, x, y, epochs=5, batch_size=32, learning_rate=0.1, shuffle=True):
# #         if type(x) == list:
# #             x = np.array(x)
# #         if type(y) == list:
# #             y = np.array(y)
# #         self.__model.train(x.astype(bool), y.astype(np.int32))
    
# #         for epoch in range(epochs):
# #             pass

    
# #     def predict(self, x):
# #         if type(x) == list:
# #             x = np.array(x)
# #         self.__model.predict(x.astype(bool))
    
# #     def clear(self):
# #         self.__model.clear()