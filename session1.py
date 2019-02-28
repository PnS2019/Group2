import numpy as np
from tensorflow.keras import backend as K

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

a = K.placeholder(shape=(5, ))
b = K.placeholder(shape=(5, ))
c = K.placeholder(shape=(5, ))

e = a**2 + b**2 + c**2 + 2 * b * c

x = K.placeholder(shape=())

tanh = (K.exp(x) - K.exp(-x)) / (K.exp(x) + K.exp(-x))
tanh_function = K.function(inputs=[x], outputs=(tanh,))

grad = K.gradients(loss=tanh, variables=[x])
grad_function = K.function(inputs=[x], outputs=(grad[0],))

values = [-100, 10, 0, 10, 100]

for value in values:
    print("x = {}:".format(value))
    print("\ttanh = {}".format(tanh_function([value])))
    print("\tgrad = {}".format(grad_function([value])))
    print()
