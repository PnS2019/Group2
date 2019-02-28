import numpy as np
from tensorflow.keras import backend as K

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
"""
# Exercise 1
a = K.placeholder(shape=(5,))
b = K.placeholder(shape=(5,))
c = K.placeholder(shape=(5,))

e = a**2 + b**2 + c**2 + 2 * b * c
fun = K.function(inputs=[a, b, c], outputs=(e,))

# Exercise 2
x = K.placeholder(shape=())

tanh = (K.exp(x) - K.exp(-x)) / (K.exp(x) + K.exp(-x))
tanh_function = K.function(inputs=[x], outputs=(tanh,))

grad = K.gradients(loss=tanh, variables=[x])
grad_function = K.function(inputs=[x], outputs=(grad[0],))

values = [-100, 10, 0, 10, 100]

print("Exercise 2:")
for value in values:
    print("x = {}:".format(value))
    print("\ttanh = {}".format(tanh_function([value])))
    print("\tgrad = {}".format(grad_function([value])))
    print()

# Exercise 3
w = K.placeholder(shape=(2,))
b = K.placeholder(shape=(1,))
x = K.placeholder(shape=(2,))

z = 1 / (1 + K.exp(w[0] * x[0] + w[1] * x[1] + b))

f = K.function(inputs=[w, b, x], outputs=(z,))

v_w = np.ones(shape=(2,))
v_b = np.ones(shape=(1,))
v_x = np.ones(shape=(2,))

print("Exercise 3:")
for mul in range(10):
    print("mul = {} : {}".format(mul, f([mul * v_w, mul * v_b, mul * v_x])))
"""

# Exercise 4

x = K.placeholder(shape=())
n = 4

eq = 0

variables = []
for i in range(n + 1):
    variables.append(K.placeholder(shape=()))
    eq += variables[-1] * x**i

grad = K.gradients(loss=eq, variables=variables)

print("Exercise 4:")
print(grad)
