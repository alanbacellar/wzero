import numpy as np
import time

class A:
    def __init__(self, x):
        self.x = x

class B(A, object):
    def __new__(cls, a=None, b=None, c=None):
        if a is None or b is None or c is None:
            def b_generator(a=a, b=b, c=c):
                return B(a,b,c)
            return b_generator
        obj = super().__new__(cls)
        obj.__init__(a, b, c)
        return obj
        
    def __init__(self, a, b, c):
        super().__init__(a)
        self.b = b
        self.c = c





# x = np.random.random((10000, 784))

# def t():
#     w1 = np.random.random((784, 128))
#     w2 = np.random.random((128, 10))
#     b1 = np.random.random((128))
#     b2 = np.random.random((10))
#     a = time.time()
#     out = np.maximum(x @ w1 + b1, 0) @ w2 + b2
#     return time.time()-a, out

# tt, out = t()
# print(tt)
    
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers  import Input, Dense

# model = Sequential((
#     Input((784,)),
#     Dense(128, activation='relu'),
#     Dense(10)
# ))

# model.summary()

# model.compile()
# a = time.time()
# pred = model.predict(x)
# print(time.time() - a)

# t = 0
# for i in range(100):
#     a = time.time()
#     pred = model.predict(x)
#     t += time.time() - a
# print(t/100)

